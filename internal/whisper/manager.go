// Package whisper manages the whisper-server child process (Manager) and
// provides an HTTP client for its API (Client).
package whisper

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/tekmanic/whisper-oai/internal/config"
)

// Manager supervises a whisper-server child process: it builds the argv,
// launches the binary, waits for readiness, and handles shutdown.
type Manager struct {
	cfg  config.WhisperConfig
	log  *slog.Logger
	cli  *Client
	cmd  *exec.Cmd
	done chan struct{}
	mu   sync.Mutex

	// Readiness/shutdown tunables. NewManager sets production defaults;
	// tests may override them.
	healthPollInterval time.Duration
	healthTimeout      time.Duration
	stopGrace          time.Duration
}

// NewManager builds a manager for the given whisper-server configuration.
// It does not start the process.
func NewManager(cfg config.WhisperConfig, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	return &Manager{
		cfg:                cfg,
		log:                log,
		cli:                NewClient("http://" + net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port))),
		healthPollInterval: 500 * time.Millisecond,
		healthTimeout:      120 * time.Second,
		stopGrace:          10 * time.Second,
	}
}

// Args returns the argv (without the binary path) that Start will use, e.g.
//
//	[-m <model> --flash-attn -dev 0 --host 127.0.0.1 --port 8080 -t 8]
//
// Order: -m, --flash-attn (only if FlashAttn), -dev (always, default 0),
// --host, --port, -t (only if Threads > 0), -l (only if Language is set and
// != "auto"), then ExtraArgs.
func (m *Manager) Args() []string {
	args := []string{"-m", m.cfg.Model}
	if m.cfg.FlashAttn {
		args = append(args, "--flash-attn")
	}
	args = append(args, "-dev", strconv.Itoa(m.cfg.Device))
	args = append(args, "--host", m.cfg.Host, "--port", strconv.Itoa(m.cfg.Port))
	if m.cfg.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(m.cfg.Threads))
	}
	if m.cfg.Language != "" && m.cfg.Language != "auto" {
		args = append(args, "-l", m.cfg.Language)
	}
	args = append(args, m.cfg.ExtraArgs...)
	return args
}

// Start launches whisper-server as a child process and blocks, polling
// GET /health every 500ms (up to ~120s) until it reports ready (200
// {"status":"ok"}). It returns an error if the process exits early, the
// readiness timeout is hit, or ctx is cancelled. The process stdout/stderr
// are piped to the logger.
func (m *Manager) Start(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cmd != nil {
		return fmt.Errorf("whisper-server is already running")
	}

	args := m.Args()
	m.log.Info("starting whisper-server", "bin", m.cfg.Bin, "args", args)

	cmd := exec.Command(m.cfg.Bin, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("pipe whisper-server stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("pipe whisper-server stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start whisper-server: %w", err)
	}

	done := make(chan struct{})
	m.cmd = cmd
	m.done = done

	go m.pump(stdout, "stdout")
	go m.pump(stderr, "stderr")
	go func() {
		waitErr := cmd.Wait()
		close(done)
		m.log.Info("whisper-server process exited", "error", waitErr)
	}()

	deadline := time.Now().Add(m.healthTimeout)
	for {
		if err := m.cli.Health(ctx); err == nil {
			m.log.Info("whisper-server is ready", "url", m.URL())
			return nil
		}
		select {
		case <-done:
			m.resetLocked()
			return fmt.Errorf("whisper-server exited before becoming ready")
		case <-ctx.Done():
			m.stopLocked()
			return fmt.Errorf("start cancelled: %w", ctx.Err())
		case <-time.After(m.healthPollInterval):
		}
		if time.Now().After(deadline) {
			m.stopLocked()
			return fmt.Errorf("whisper-server did not become ready within %s", m.healthTimeout)
		}
	}
}

// Stop sends SIGTERM, waits up to 10s for exit, then SIGKILL. It is
// idempotent: stopping a manager that is not running is a no-op.
func (m *Manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.stopLocked()
}

// stopLocked stops the child process. The caller must hold m.mu.
func (m *Manager) stopLocked() error {
	if m.cmd == nil || m.cmd.Process == nil {
		return nil
	}
	select {
	case <-m.done:
		m.resetLocked()
		return nil
	default:
	}
	if err := m.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		// The process is already gone; wait for Wait to finish.
		<-m.done
		m.resetLocked()
		return nil
	}
	select {
	case <-m.done:
	case <-time.After(m.stopGrace):
		if err := m.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("kill whisper-server: %w", err)
		}
		<-m.done
	}
	m.resetLocked()
	return nil
}

// resetLocked clears process state. The caller must hold m.mu.
func (m *Manager) resetLocked() {
	m.cmd = nil
	m.done = nil
}

// Health returns nil if whisper-server /health is 200.
func (m *Manager) Health(ctx context.Context) error {
	return m.cli.Health(ctx)
}

// URL returns the base URL (http://host:port) of the whisper-server backend.
func (m *Manager) URL() string {
	return "http://" + net.JoinHostPort(m.cfg.Host, strconv.Itoa(m.cfg.Port))
}

// Running reports whether the child process is alive.
func (m *Manager) Running() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cmd == nil || m.done == nil {
		return false
	}
	select {
	case <-m.done:
		return false
	default:
		return true
	}
}

// pump streams r to the logger, one line at a time.
func (m *Manager) pump(r io.Reader, stream string) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		m.log.Info(sc.Text(), "stream", stream)
	}
}
