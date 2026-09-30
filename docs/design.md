# Design — whisper-oai (package & type contracts)

Single source of truth for the implementation. Subagents MUST follow these
signatures exactly so the packages compile together. Go 1.22, module
`github.com/tekmanic/whisper-oai`. Only stdlib + `spf13/cobra` + `spf13/viper`.
Logging via `log/slog`.

## Layout
```
cmd/whisper-oai/main.go          package main
internal/cli/root.go             package cli
internal/cli/serve.go
internal/cli/config.go
internal/cli/version.go
internal/config/config.go        package config
internal/config/config_test.go
internal/whisper/manager.go      package whisper
internal/whisper/manager_test.go
internal/whisper/client.go
internal/whisper/client_test.go
internal/server/server.go        package server
internal/server/handlers.go
internal/server/auth.go
internal/server/errors.go
internal/server/server_test.go
internal/audio/extract.go        package audio
internal/audio/extract_test.go
```

## Dependency direction
`cli` → `server`, `whisper`, `config`
`server` → `whisper`, `config`, `audio`
`whisper` → `config`
`audio` → (stdlib only)
`config` → (stdlib, viper)

---

## package config

```go
type Config struct {
    Server  ServerConfig  `mapstructure:"server"`
    Whisper WhisperConfig `mapstructure:"whisper"`
    Audio   AudioConfig   `mapstructure:"audio"`
    Log     LogConfig     `mapstructure:"log"`
}

type ServerConfig struct {
    Host      string `mapstructure:"host"`       // default "0.0.0.0"
    Port      int    `mapstructure:"port"`       // default 8000
    APIKey    string `mapstructure:"api_key"`    // default "" (no auth)
    ModelName string `mapstructure:"model_name"` // default "whisper-1"
}

type WhisperConfig struct {
    Bin       string   `mapstructure:"bin"`        // default /opt/whisper.cpp/build/bin/whisper-server
    Model     string   `mapstructure:"model"`      // default /opt/whisper.cpp/models/ggml-large-v3-turbo.bin
    Device    int      `mapstructure:"device"`     // default 0
    FlashAttn bool     `mapstructure:"flash_attn"` // default true
    Host      string   `mapstructure:"host"`       // default "127.0.0.1" (loopback only)
    Port      int      `mapstructure:"port"`       // default 8080
    Threads   int      `mapstructure:"threads"`    // default 0 (auto)
    Language  string   `mapstructure:"language"`   // default "auto"
    ExtraArgs []string `mapstructure:"extra_args"` // default nil
}

type LogConfig struct {
    Level string `mapstructure:"level"` // default "info"
}

type AudioConfig struct {
    FfmpegBin   string `mapstructure:"ffmpeg_bin"`   // default "ffmpeg"
    TempDir     string `mapstructure:"temp_dir"`     // default "" (os.TempDir)
    MaxUploadMB int    `mapstructure:"max_upload_mb"` // default 2048
}

// Load reads config from (in precedence order): explicit flags (set by caller
// via SetFlag), env vars (WHISPER_OAI_*), the file at `path` (if non-empty and
// exists), then built-in defaults. Returns a fully-populated *Config.
func Load(path string) (*Config, error)

// SetFlag records a CLI flag override (key = dotted, e.g. "server.port", value
// string). Call before Load. Safe to call with empty path.
func SetFlag(key, value string)

// Validate checks bin/model are non-empty, ports in [1,65535], host non-empty.
// It does NOT stat the bin/model files (so tests pass without a real install);
// the serve command logs a warning if they are missing.
func (c *Config) Validate() error

// ListenAddr returns "host:port" for the public OpenAI-compatible server.
func (c *Config) ListenAddr() string

// WhisperURL returns "http://host:port" for the whisper-server backend.
func (c *Config) WhisperURL() string
```

Env var mapping: `WHISPER_OAI_SERVER_PORT`, `WHISPER_OAI_WHISPER_BIN`, etc.
(viper `SetEnvPrefix("WHISPER_OAI")`, `SetEnvKeyReplacer` mapping `.`→`_`,
`AutomaticEnv`).

---

## package whisper

### Manager (process lifecycle)

```go
type Manager struct { /* unexported: cfg, cmd, log, done chan, mu */ }

// NewManager builds a manager. Does not start the process.
func NewManager(cfg config.WhisperConfig, log *slog.Logger) *Manager

// Args returns the argv (without the binary path) that Start will use, e.g.
//   [-m <model> --flash-attn -dev 0 --host 127.0.0.1 --port 8080 -t 8]
// Order: -m, then --flash-attn (only if FlashAttn), then -dev (only if Device
// set or always), then --host, --port, then -t (only if Threads>0), then
// -l (only if Language != "" and != "auto"), then ExtraArgs.
// NOTE: always emit `-dev <Device>` (default 0) to match the reference command.
func (m *Manager) Args() []string

// Start launches whisper-server as a child process and blocks (polling
// GET /health every 500ms, up to ~120s) until it reports ready
// (200 {"status":"ok"}). Returns error if the process exits early or the
// readiness timeout is hit. The process stdout/stderr are piped to the logger.
func (m *Manager) Start(ctx context.Context) error

// Stop sends SIGTERM, waits up to 10s for exit, then SIGKILL. Idempotent.
func (m *Manager) Stop() error

// Health returns nil if whisper-server /health is 200.
func (m *Manager) Health(ctx context.Context) error

// URL returns the base URL (http://host:port).
func (m *Manager) URL() string

// Running reports whether the child process is alive.
func (m *Manager) Running() bool
```

### Client (HTTP to whisper-server)

```go
type Client struct { /* unexported: baseURL, http */ }

func NewClient(baseURL string) *Client

type InferRequest struct {
    File           io.Reader // audio bytes
    Filename       string    // e.g. "audio.wav"
    Language       string    // "" = server default
    Prompt         string
    Temperature    float64
    Translate      bool
    ResponseFormat string    // "json"|"text"|"srt"|"vtt"|"verbose_json"
}

type Segment struct {
    ID               int     `json:"id"`
    Text             string  `json:"text"`
    Start            float64 `json:"start"`
    End              float64 `json:"end"`
    Tokens           []int64 `json:"tokens,omitempty"`
    Temperature      float64 `json:"temperature"`
    AvgLogprob       float64 `json:"avg_logprob"`
    NoSpeechProb     float64 `json:"no_speech_prob"`
    CompressionRatio float64 `json:"compression_ratio"`
    Seek             int     `json:"seek"`
}

type InferResponse struct {
    Text        string    // populated for json/verbose_json
    Language    string    // populated for verbose_json
    Duration    float64   // populated for verbose_json
    Segments    []Segment // populated for verbose_json
    RawBody     []byte    // populated for text/srt/vtt (passthrough)
    ContentType string    // content type of the upstream response
}

// Infer POSTs a multipart request to /inference. The multipart field names are
// EXACTLY: file, temperature, response_format, language, prompt, translate.
// (translate is "true"/"false" lowercase.) It parses the response per
// ResponseFormat and returns a populated *InferResponse. Non-2xx → error with
// the upstream body included.
func (c *Client) Infer(ctx context.Context, req InferRequest) (*InferResponse, error)

// Health returns nil if /health is 200.
func (c *Client) Health(ctx context.Context) error
```

whisper-server `verbose_json` upstream shape (parse this):
```json
{"task":"transcribe","language":"english","duration":3.32,"text":"...",
 "segments":[{"id":0,"text":"...","start":0.0,"end":3.32,"tokens":[1,2],
 "temperature":0.0,"avg_logprob":-0.28,"no_speech_prob":0.01}]}
```
Map to `InferResponse`; set `CompressionRatio=0` and `Seek=0` (whisper omits
them). Plain `json` upstream shape: `{"text":"..."}`.

---

## package audio

Audio extraction for video uploads (mp4, mov, mkv, webm, ...). whisper-server
decodes raw audio (wav/mp3/ogg/flac/opus) but NOT video containers, so the
proxy strips the audio track with ffmpeg before forwarding.

```go
// IsVideo reports whether the uploaded file is a video container that needs
// audio extraction, based on filename extension or Content-Type.
// Video extensions: mp4, m4v, mov, mkv, webm, avi, wmv, flv, mpg, mpeg,
// 3gp, ts, m2ts, ogv. Content-Type "video/*" also matches.
func IsVideo(filename, contentType string) bool

// Spool copies the uploaded bytes to a temp file so large uploads stream to
// disk (not memory) and ffmpeg can address them by path. The temp file is
// created in tempDir ("" → os.TempDir) with a unique name. Returns the path;
// the caller MUST os.Remove it.
func Spool(src io.Reader, filename, tempDir string) (string, error)

// Extract runs ffmpeg to pull the audio track out of a video file and write
// a 16 kHz mono PCM WAV (the format whisper.cpp consumes natively).
// Command: <ffmpegBin> -nostdin -hide_banner -loglevel error -y -i <srcPath>
//          -vn -ac 1 -ar 16000 -c:a pcm_s16le <dst.wav>
// The output .wav is created in tempDir ("" → os.TempDir) with a unique name.
// Returns the wav path; the caller MUST os.Remove it. Non-zero exit or a
// missing ffmpeg binary → error (include ffmpeg stderr in the error).
func Extract(ctx context.Context, srcPath, ffmpegBin, tempDir string, log *slog.Logger) (string, error)
```

---

## package server

```go
type Server struct { /* unexported: cfg, mgr, cli, http, log */ }

// New wires the HTTP server. cli is built from mgr.URL().
func New(cfg *config.Config, mgr *whisper.Manager, log *slog.Logger) *Server

// Handler returns the http.Handler (mux) — exposed for tests.
func (s *Server) Handler() http.Handler

// Start runs http.Server on cfg.ListenAddr(); blocks until ctx is cancelled
// or a fatal error. On ctx cancel it calls Shutdown.
func (s *Server) Start(ctx context.Context) error

// Shutdown gracefully stops the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error
```

Routes (mux):
- `POST /v1/audio/transcriptions` → handleTranscription (translate=false)
- `POST /v1/audio/translations`   → handleTranslation  (translate=true)
- `GET  /v1/models`               → handleModels
- `GET  /v1/models/{model}`       → handleModel (404 if != cfg.Server.ModelName)
- `GET  /healthz`                 → 200 `{"status":"ok"}` (proxy liveness)

### handlers.go
- Parse OpenAI multipart (`file`, `model`, `language`, `prompt`,
  `response_format`, `temperature`). Validate `file` present (400 if not).
  `model` must equal cfg.Server.ModelName (400 if not). `response_format`
  default `json`; accept json/text/srt/verbose_json/vtt (400 otherwise).
- Enforce `audio.max_upload_mb` via `http.MaxBytesReader` before parsing
  (413 when exceeded).
- Spool the upload to a temp file (`audio.Spool`) so large uploads stream to
  disk. If `audio.IsVideo(filename, contentType)`, run `audio.Extract`
  (ffmpeg) to get a 16 kHz mono WAV and forward THAT (filename = the wav's
  basename); otherwise forward the spooled file as-is. Extraction failure →
  502 with OpenAI error shape.
- Map to `whisper.InferRequest`, call `cli.Infer`.
- Map `whisper.InferResponse` back to the OpenAI shape for the requested
  `response_format`:
  - `json` → `{"text": ...}`
  - `verbose_json` → `{task, language, duration, text, segments[]}` where
    task = "transcribe"|"translate", segments use the OpenAI Segment field set
    (id, seek, start, end, text, tokens, temperature, avg_logprob,
    compression_ratio, no_speech_prob).
  - `text`/`srt`/`vtt` → passthrough RawBody with the upstream ContentType
    (but force `text/plain; charset=utf-8` for `text`).
- Upstream error → 502 with OpenAI error shape.

### auth.go
```go
// authMiddleware returns a middleware that, when apiKey != "", requires
// "Authorization: Bearer <apiKey>" (401 otherwise). When apiKey == "", it is a
// no-op passthrough.
func authMiddleware(apiKey string, next http.Handler) http.Handler
```

### errors.go
```go
// writeError writes the OpenAI error shape with the given HTTP status.
// errType is one of: invalid_request_error, authentication_error, server_error,
// service_unavailable. param may be "".
func writeError(w http.ResponseWriter, status int, errType, message, param string)
```
Body: `{"error":{"message":...,"type":...,"param":...,"code":null}}` (omit
`param`/`code` when empty/null — use a struct with omitempty).

---

## package cli

```go
// Execute builds the root command and runs it; returns the error (main maps
// this to os.Exit(1)).
func Execute() error

// NewRootCmd returns the root cobra command (for tests).
func NewRootCmd() *cobra.Command
```

Root command `whisper-oai`, persistent flag `--config` (string, default
`config/whisper-oai.yaml`). Subcommands:
- `serve` — flags: `--host`, `--port`, `--api-key`, `--model-name`,
  `--whisper-bin`, `--whisper-model`, `--whisper-host`, `--whisper-port`,
  `--device`, `--threads`, `--language`, `--no-flash-attn`. Loads config
  (flags override), validates, builds logger, creates Manager, starts it,
  creates Server, runs it; on SIGINT/SIGTERM shuts down server then manager.
- `config init` — writes the example config to `--config` path (or
  `config/whisper-oai.yaml`); refuses to overwrite unless `--force`.
- `config get <key>` — prints a single value (dotted key).
- `config set <key> <value>` — sets a value in the config file (creates if
  missing).
- `config path` — prints the resolved config file path.
- `version` — prints version (ldflags-injected `var version string`, default
  "dev").

Flag→config key mapping for `serve` (call config.SetFlag before Load):
`--host`→server.host, `--port`→server.port, `--api-key`→server.api_key,
`--model-name`→server.model_name, `--whisper-bin`→whisper.bin,
`--whisper-model`→whisper.model, `--whisper-host`→whisper.host,
`--whisper-port`→whisper.port, `--device`→whisper.device,
`--threads`→whisper.threads, `--language`→whisper.language,
`--no-flash-attn`→whisper.flash_attn=false.

---

## Conventions
- Errors: wrap with `fmt.Errorf("...: %w", err)`.
- No global mutable state; pass `*slog.Logger`.
- Table-driven tests; use `net/http/httptest` for server + whisper client
  tests (fake whisper-server). No real GPU/binary needed for tests.
- `gofmt` clean; `go vet` clean.
- Every exported symbol has a doc comment.
