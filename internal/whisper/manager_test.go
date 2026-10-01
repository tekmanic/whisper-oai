package whisper

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/tekmanic/whisper-oai/internal/config"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// writeScript writes an executable shell script that stands in for the
// whisper-server binary (it just needs to be a process we can start/kill).
func writeScript(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "fake-server.sh")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatalf("write fake server script: %v", err)
	}
	return p
}

// hostPort splits an httptest URL into host and port.
func hostPort(t *testing.T, rawURL string) (string, int) {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse url %q: %v", rawURL, err)
	}
	host, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatalf("split host:port %q: %v", u.Host, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("parse port %q: %v", portStr, err)
	}
	return host, port
}

// healthServer starts a fake whisper-server that answers GET /health with the
// given status.
func healthServer(t *testing.T, status int) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	t.Cleanup(ts.Close)
	return ts
}

func TestArgs(t *testing.T) {
	tests := []struct {
		name string
		cfg  config.WhisperConfig
		want []string
	}{
		{
			name: "defaults match reference command",
			cfg: config.WhisperConfig{
				Model:     "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin",
				Device:    0,
				FlashAttn: true,
				Host:      "127.0.0.1",
				Port:      8080,
			},
			want: []string{
				"-m", "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin",
				"--flash-attn",
				"-dev", "0",
				"--host", "127.0.0.1",
				"--port", "8080",
			},
		},
		{
			name: "threads and language",
			cfg: config.WhisperConfig{
				Model:     "/models/base.bin",
				FlashAttn: true,
				Device:    1,
				Host:      "0.0.0.0",
				Port:      9090,
				Threads:   8,
				Language:  "en",
			},
			want: []string{
				"-m", "/models/base.bin",
				"--flash-attn",
				"-dev", "1",
				"--host", "0.0.0.0",
				"--port", "9090",
				"-t", "8",
				"-l", "en",
			},
		},
		{
			name: "no flash attn, auto language omitted",
			cfg: config.WhisperConfig{
				Model:     "/m.bin",
				FlashAttn: false,
				Device:    0,
				Host:      "127.0.0.1",
				Port:      8080,
				Language:  "auto",
			},
			want: []string{
				"-m", "/m.bin",
				"-dev", "0",
				"--host", "127.0.0.1",
				"--port", "8080",
			},
		},
		{
			name: "empty language omitted",
			cfg: config.WhisperConfig{
				Model:    "/m.bin",
				Host:     "127.0.0.1",
				Port:     8080,
				Language: "",
			},
			want: []string{
				"-m", "/m.bin",
				"-dev", "0",
				"--host", "127.0.0.1",
				"--port", "8080",
			},
		},
		{
			name: "extra args appended last",
			cfg: config.WhisperConfig{
				Model:     "/m.bin",
				FlashAttn: true,
				Host:      "127.0.0.1",
				Port:      8080,
				ExtraArgs: []string{"-p", "2", "--no-gpu"},
			},
			want: []string{
				"-m", "/m.bin",
				"--flash-attn",
				"-dev", "0",
				"--host", "127.0.0.1",
				"--port", "8080",
				"-p", "2",
				"--no-gpu",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewManager(tt.cfg, testLogger())
			got := m.Args()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Args() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestURL(t *testing.T) {
	m := NewManager(config.WhisperConfig{Host: "127.0.0.1", Port: 8080}, testLogger())
	if got := m.URL(); got != "http://127.0.0.1:8080" {
		t.Errorf("URL() = %q, want http://127.0.0.1:8080", got)
	}
}

func TestManagerHealth(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		ts := healthServer(t, http.StatusOK)
		host, port := hostPort(t, ts.URL)
		m := NewManager(config.WhisperConfig{Host: host, Port: port}, testLogger())
		if err := m.Health(context.Background()); err != nil {
			t.Fatalf("Health: %v", err)
		}
	})
	t.Run("not ready", func(t *testing.T) {
		ts := healthServer(t, http.StatusServiceUnavailable)
		host, port := hostPort(t, ts.URL)
		m := NewManager(config.WhisperConfig{Host: host, Port: port}, testLogger())
		if err := m.Health(context.Background()); err == nil {
			t.Fatal("Health: expected error for 503")
		}
	})
}

func TestStartStop(t *testing.T) {
	ts := healthServer(t, http.StatusOK)
	host, port := hostPort(t, ts.URL)
	bin := writeScript(t, "sleep 30\n")

	m := NewManager(config.WhisperConfig{
		Bin:       bin,
		Model:     "/models/fake.bin",
		FlashAttn: true,
		Host:      host,
		Port:      port,
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 5 * time.Second

	if err := m.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if !m.Running() {
		t.Fatal("Running() = false after Start, want true")
	}
	if err := m.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if m.Running() {
		t.Fatal("Running() = true after Stop, want false")
	}
	// Stop is idempotent.
	if err := m.Stop(); err != nil {
		t.Fatalf("second Stop: %v", err)
	}
	// The manager can be started again after Stop.
	if err := m.Start(context.Background()); err != nil {
		t.Fatalf("restart: %v", err)
	}
	if err := m.Stop(); err != nil {
		t.Fatalf("Stop after restart: %v", err)
	}
}

func TestStartProcessExitsEarly(t *testing.T) {
	ts := healthServer(t, http.StatusServiceUnavailable)
	host, port := hostPort(t, ts.URL)
	bin := writeScript(t, "exit 1\n")

	m := NewManager(config.WhisperConfig{
		Bin:   bin,
		Model: "/m.bin",
		Host:  host,
		Port:  port,
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 5 * time.Second

	if err := m.Start(context.Background()); err == nil {
		t.Fatal("Start: expected error when the process exits early")
	}
	if m.Running() {
		t.Fatal("Running() = true after failed Start, want false")
	}
}

func TestStartTimeout(t *testing.T) {
	ts := healthServer(t, http.StatusServiceUnavailable)
	host, port := hostPort(t, ts.URL)
	bin := writeScript(t, "sleep 30\n")

	m := NewManager(config.WhisperConfig{
		Bin:   bin,
		Model: "/m.bin",
		Host:  host,
		Port:  port,
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 300 * time.Millisecond
	m.stopGrace = 200 * time.Millisecond

	start := time.Now()
	if err := m.Start(context.Background()); err == nil {
		t.Fatal("Start: expected readiness timeout error")
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("Start took too long: %v", elapsed)
	}
	if m.Running() {
		t.Fatal("Running() = true after timeout; child was not cleaned up")
	}
}

func TestStopKillsStuckProcess(t *testing.T) {
	ts := healthServer(t, http.StatusOK)
	host, port := hostPort(t, ts.URL)
	// Ignore SIGTERM so Stop must escalate to SIGKILL. The trailing `wait`
	// keeps the shell (not just a forked sleep) as the process we signal.
	bin := writeScript(t, "trap '' TERM\nsleep 30 &\nwait\n")

	m := NewManager(config.WhisperConfig{
		Bin:   bin,
		Model: "/m.bin",
		Host:  host,
		Port:  port,
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 5 * time.Second
	m.stopGrace = 300 * time.Millisecond

	if err := m.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := m.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if m.Running() {
		t.Fatal("Running() = true after Stop of a stuck process, want false")
	}
}

func TestStartAlreadyRunning(t *testing.T) {
	ts := healthServer(t, http.StatusOK)
	host, port := hostPort(t, ts.URL)
	bin := writeScript(t, "sleep 30\n")

	m := NewManager(config.WhisperConfig{
		Bin:   bin,
		Model: "/m.bin",
		Host:  host,
		Port:  port,
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 5 * time.Second

	if err := m.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer m.Stop()
	if err := m.Start(context.Background()); err == nil {
		t.Fatal("second Start: expected error while already running")
	}
}

func TestStartMissingBinary(t *testing.T) {
	m := NewManager(config.WhisperConfig{
		Bin:   "/nonexistent/whisper-server",
		Model: "/m.bin",
		Host:  "127.0.0.1",
		Port:  8080,
	}, testLogger())
	if err := m.Start(context.Background()); err == nil {
		t.Fatal("Start: expected error for missing binary")
	}
	if m.Running() {
		t.Fatal("Running() = true after failed Start, want false")
	}
}

func TestURLRemote(t *testing.T) {
	m := NewManager(config.WhisperConfig{RemoteURL: "http://remote.example.com:8080/"}, testLogger())
	if got := m.URL(); got != "http://remote.example.com:8080" {
		t.Errorf("URL() = %q, want http://remote.example.com:8080", got)
	}
}

func TestStartRemoteNoProcess(t *testing.T) {
	ts := healthServer(t, http.StatusOK)
	m := NewManager(config.WhisperConfig{
		RemoteURL: ts.URL,
		Bin:       "/nonexistent/whisper-server",
		Model:     "/m.bin",
	}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 2 * time.Second

	if err := m.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if m.Running() {
		t.Fatal("Running() = true in remote mode, want false")
	}
	if err := m.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
}

func TestStartRemoteTimeout(t *testing.T) {
	ts := healthServer(t, http.StatusServiceUnavailable)
	m := NewManager(config.WhisperConfig{RemoteURL: ts.URL}, testLogger())
	m.healthPollInterval = 20 * time.Millisecond
	m.healthTimeout = 200 * time.Millisecond

	if err := m.Start(context.Background()); err == nil {
		t.Fatal("Start: expected readiness timeout in remote mode")
	}
}
