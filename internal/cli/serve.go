package cli

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/tekmanic/whisper-oai/internal/config"
	"github.com/tekmanic/whisper-oai/internal/server"
	"github.com/tekmanic/whisper-oai/internal/whisper"
)

// newServeCmd returns the serve subcommand.
func newServeCmd() *cobra.Command {
	var (
		host         string
		port         int
		apiKey       string
		verbose      bool
		modelName    string
		whisperBin   string
		whisperModel string
		remoteURL    string
		whisperHost  string
		whisperPort  int
		device       int
		threads      int
		language     string
		noFlashAttn  bool
	)

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start whisper-server and the OpenAI-compatible API",
		Long: `Starts the whisper-server child process, waits until it reports ready,
then serves the OpenAI-compatible API on server.host:server.port.

If whisper.remote_url is set (or --whisper-remote-url is provided), whisper-oai
uses that backend directly and does not start a local whisper-server process.

Configuration is loaded from the --config file; any flag given on the
command line overrides the file and environment variables. The process
shuts down gracefully on SIGINT/SIGTERM.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			f := cmd.Flags()
			if f.Changed("host") {
				config.SetFlag("server.host", host)
			}
			if f.Changed("port") {
				config.SetFlag("server.port", strconv.Itoa(port))
			}
			if f.Changed("api-key") {
				config.SetFlag("server.api_key", apiKey)
			}
			if f.Changed("model-name") {
				config.SetFlag("server.model_name", modelName)
			}
			if f.Changed("whisper-bin") {
				config.SetFlag("whisper.bin", whisperBin)
			}
			if f.Changed("whisper-model") {
				config.SetFlag("whisper.model", whisperModel)
			}
			if f.Changed("whisper-remote-url") {
				config.SetFlag("whisper.remote_url", remoteURL)
			}
			if f.Changed("whisper-host") {
				config.SetFlag("whisper.host", whisperHost)
			}
			if f.Changed("whisper-port") {
				config.SetFlag("whisper.port", strconv.Itoa(whisperPort))
			}
			if f.Changed("device") {
				config.SetFlag("whisper.device", strconv.Itoa(device))
			}
			if f.Changed("threads") {
				config.SetFlag("whisper.threads", strconv.Itoa(threads))
			}
			if f.Changed("language") {
				config.SetFlag("whisper.language", language)
			}
			if f.Changed("no-flash-attn") {
				config.SetFlag("whisper.flash_attn", "false")
			}

			cfgPath, err := f.GetString("config")
			if err != nil {
				return fmt.Errorf("read --config flag: %w", err)
			}
			cfg, err := config.Load(cfgPath)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}
			if err := cfg.Validate(); err != nil {
				return fmt.Errorf("invalid config: %w", err)
			}

			logLevel := cfg.Log.Level
			if verbose {
				logLevel = "debug"
			}
			log := newLogger(logLevel)
			if verbose && cfg.Log.Level != "debug" {
				log.Info("verbose logging enabled via --verbose", "configured_level", cfg.Log.Level, "effective_level", logLevel)
			}
			log.Debug("loaded configuration",
				"listen_addr", cfg.ListenAddr(),
				"whisper_url", cfg.WhisperURL(),
				"whisper_remote", cfg.Whisper.RemoteURL != "",
				"model_name", cfg.Server.ModelName,
				"auth_enabled", cfg.Server.APIKey != "",
				"max_upload_mb", cfg.Audio.MaxUploadMB,
				"ffmpeg_bin", cfg.Audio.FfmpegBin,
			)

			if cfg.Whisper.RemoteURL == "" {
				// Validate does not stat these paths; warn if they are missing.
				for _, p := range []string{cfg.Whisper.Bin, cfg.Whisper.Model} {
					if _, err := os.Stat(p); err != nil {
						log.Warn("configured path does not exist", "path", p)
					}
				}
			}
			// ffmpeg is only needed for video uploads; warn early if absent.
			if _, err := exec.LookPath(cfg.Audio.FfmpegBin); err != nil {
				log.Warn("ffmpeg not found; video uploads (mp4, mov, mkv, ...) will fail until it is installed",
					"bin", cfg.Audio.FfmpegBin)
			}

			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			defer stop()

			mgr := whisper.NewManager(cfg.Whisper, log)
			if cfg.Whisper.RemoteURL == "" {
				log.Info("starting whisper-server", "bin", cfg.Whisper.Bin, "model", cfg.Whisper.Model)
			} else {
				log.Info("using remote whisper-server", "url", mgr.URL())
			}
			log.Info("waiting for whisper-server readiness", "url", mgr.URL())
			if err := mgr.Start(ctx); err != nil {
				return fmt.Errorf("start whisper-server: %w", err)
			}
			log.Info("whisper-server ready", "url", mgr.URL())

			srv := server.New(cfg, mgr, log)
			log.Info("serving OpenAI-compatible API", "addr", cfg.ListenAddr())

			errCh := make(chan error, 1)
			go func() {
				errCh <- srv.Start(ctx)
			}()

			select {
			case <-ctx.Done():
				log.Info("shutdown signal received")
				sc, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				defer cancel()
				if err := srv.Shutdown(sc); err != nil && !errors.Is(err, http.ErrServerClosed) {
					log.Error("api server shutdown failed", "error", err)
				}
				if err := mgr.Stop(); err != nil {
					log.Error("stop whisper-server failed", "error", err)
				}
				log.Info("shutdown complete")
				return nil
			case err := <-errCh:
				if err := mgr.Stop(); err != nil {
					log.Error("stop whisper-server failed", "error", err)
				}
				if err != nil {
					return fmt.Errorf("serve: %w", err)
				}
				return nil
			}
		},
	}

	fs := cmd.Flags()
	fs.StringVar(&host, "host", "", "host/interface the API server listens on (server.host)")
	fs.IntVar(&port, "port", 0, "port the API server listens on (server.port)")
	fs.StringVar(&apiKey, "api-key", "", "require Bearer-token auth with this key (server.api_key)")
	fs.BoolVar(&verbose, "verbose", false, "enable verbose debug logging")
	fs.StringVar(&modelName, "model-name", "", "model name advertised by /v1/models (server.model_name)")
	fs.StringVar(&whisperBin, "whisper-bin", "", "path to the whisper-server binary (whisper.bin)")
	fs.StringVar(&whisperModel, "whisper-model", "", "path to the GGML model file (whisper.model)")
	fs.StringVar(&remoteURL, "whisper-remote-url", "", "remote whisper-server base URL (whisper.remote_url), e.g. http://127.0.0.1:8080")
	fs.StringVar(&whisperHost, "whisper-host", "", "host whisper-server binds to (whisper.host)")
	fs.IntVar(&whisperPort, "whisper-port", 0, "port whisper-server binds to (whisper.port)")
	fs.IntVar(&device, "device", 0, "GPU device index for whisper-server (whisper.device)")
	fs.IntVar(&threads, "threads", 0, "CPU threads for whisper-server, 0 = auto (whisper.threads)")
	fs.StringVar(&language, "language", "", "default inference language (whisper.language)")
	fs.BoolVar(&noFlashAttn, "no-flash-attn", false, "disable flash attention (whisper.flash_attn=false)")

	return cmd
}

// newLogger builds a *slog.Logger with a text handler on stderr at the given
// level ("debug", "info", "warn", "error"); unknown levels fall back to info.
func newLogger(level string) *slog.Logger {
	var lv slog.Level
	if err := lv.UnmarshalText([]byte(level)); err != nil {
		lv = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lv}))
}
