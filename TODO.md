# TODO — whisper-oai

Golang wrapper around **whisper.cpp**'s `whisper-server` that exposes an
**OpenAI-compatible** HTTP API. The Go app owns the lifecycle of
`whisper-server` (starts it, talks to it, stops it) and is the *only*
consumer of the whisper-server port — whisper-server binds to loopback
(`127.0.0.1`) so nothing else on the box can reach it.

Reference for the OpenAI API surface we mirror:
<https://github.com/openai/openai-openapi/blob/main/openapi.yaml>

---

## Status legend
- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

## 0. Scaffold (done)
- [x] `go.mod` — module `github.com/tekmanic/whisper-oai`, go 1.22, cobra + viper
- [x] `Makefile` — build/run/test/vet/fmt/tidy/install/clean
- [x] `config/whisper-oai.example.yaml` — documented example config
- [x] `.gitignore` — ignores `bin/`, real config, logs, IDE cruft
- [x] `LICENSE` — MIT
- [x] GitHub repo `tekmanic/whisper-oai` exists (empty, no refs yet)
- [x] `go.sum` — generated via `go mod tidy`

## 1. Research (done — findings captured in `docs/`)
- [x] `docs/research-openai.md` — OpenAI audio endpoints to mirror:
  - `POST /v1/audio/transcriptions` (multipart: file, model, language, prompt,
    temperature, response_format, timestamp_granularities[])
  - `POST /v1/audio/translations` (same shape, no `temperature` semantics differ)
  - `GET /v1/models` and `GET /v1/models/{model}`
- [x] `docs/research-whisper-server.md` — whisper-server endpoints to call:
  - `POST /inference` (multipart: file, temperature, **response_format**,
    language, prompt, translate, n_threads, ...)
  - `GET /health`, `GET /models`, `GET /timestamps`
  - CLI flags: `-m`, `--flash-attn`, `-dev`, `--host`, `--port`, `-t`, `-l`
- [x] Response-format mapping (OpenAI `response_format` → whisper `response_format`):
  - `json` → whisper `json`, return `.text`
  - `verbose_json` → whisper `verbose_json`, map segments to OpenAI verbose shape
  - `text` → whisper `text`
  - `srt` → whisper `srt`
  - `vtt` → whisper `vtt`
- [x] `docs/design.md` — package & type contracts (single source of truth)

## 2. Config layer — `internal/config/`
- [ ] `config.go` — `Config`, `ServerConfig`, `WhisperConfig`, `AudioConfig`,
      `LogConfig` structs with `mapstructure` tags matching the example YAML
- [ ] `Load(path string) (*Config, error)` — viper: defaults → config file → env
      (`WHISPER_OAI_*`) → flags; precedence flag > env > file > default
- [ ] `Validate()` — bin/model non-empty, ports in range, hosts non-empty
      (no stat; serve warns on missing paths)
- [ ] `ListenAddr()` / `WhisperURL()` helpers
- [ ] unit test: load example config, validate, precedence

## 3. whisper-server process manager — `internal/whisper/`
- [ ] `manager.go` — `Manager` struct wrapping `exec.Cmd`
  - `NewManager(cfg config.WhisperConfig, log *slog.Logger) *Manager`
  - `Args()` — argv: `-m <model> [--flash-attn] -dev <device> --host <host>
    --port <port> [-t N] [-l LANG] <extra_args...>` (always emit `-dev`)
  - `Start(ctx) error` — exec, pipe stdout/stderr to slog, poll `GET /health`
    every 500ms up to ~120s
  - `Stop()` — SIGTERM, wait 10s, SIGKILL fallback; idempotent
  - `Health(ctx) error`, `URL() string`, `Running() bool`
- [ ] `client.go` — `Client` for whisper-server HTTP
  - `Infer(ctx, InferRequest) (*InferResponse, error)` — multipart POST
    `/inference` with EXACT fields: file, temperature, response_format,
    language, prompt, translate ("true"/"false")
  - `InferRequest{File, Filename, Language, Prompt, Temperature, Translate,
    ResponseFormat}`
  - `InferResponse{Text, Language, Duration, Segments []Segment, RawBody,
    ContentType}` + `Segment` struct (CompressionRatio/Seek zeroed)
  - `Health(ctx) error`
- [ ] unit tests: argv construction (table-driven), health polling with a fake
      server, client Infer against an httptest server

## 3b. Audio extraction — `internal/audio/` (NEW — mp4 support)
- [ ] `extract.go` — stdlib-only package:
  - `IsVideo(filename, contentType string) bool` — video extensions
    (mp4, m4v, mov, mkv, webm, avi, wmv, flv, mpg, mpeg, 3gp, ts, m2ts, ogv)
    or `video/*` content type
  - `Spool(src io.Reader, filename, tempDir string) (string, error)` — copy
    upload to a temp file (large files stream to disk, not memory)
  - `Extract(ctx, srcPath, ffmpegBin, tempDir string, log *slog.Logger)
    (string, error)` — run ffmpeg:
    `-nostdin -hide_banner -loglevel error -y -i <src> -vn -ac 1 -ar 16000
    -c:a pcm_s16le <dst.wav>` (16 kHz mono PCM = whisper.cpp native)
- [ ] `extract_test.go` — IsVideo table-driven; Spool round-trip; Extract
      against a fake ffmpeg (scripted shell binary) — no real ffmpeg needed

## 4. OpenAI-compatible HTTP server — `internal/server/`
- [x] `server.go` — `Server` struct, `New(cfg, mgr, log)`, `Start(ctx)`,
      `Shutdown(ctx)`; routes:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
  - `GET  /v1/models`
  - `GET  /v1/models/{model}`
  - `GET  /healthz` (proxy liveness)
- [x] `handlers.go` — parse OpenAI multipart, map to whisper `InferRequest`,
      call client, map response back to OpenAI shape (json / verbose_json /
      text / srt / vtt)
- [ ] `handlers.go` — **wire in audio**: MaxBytesReader (413), Spool upload,
      IsVideo → Extract (ffmpeg) → forward wav; extraction failure → 502
- [x] `auth.go` — optional Bearer `api_key` middleware (skip when empty)
- [x] `errors.go` — OpenAI-style `{"error": {"message", "type", "code"}}`
- [x] `server_test.go` — httptest against handlers with a fake whisper-server
- [ ] `server_test.go` — add video-upload test (fake ffmpeg) + 413 test

## 5. CLI — `cmd/whisper-oai/` + `internal/cli/`
- [x] `main.go` — thin: call `cli.Execute()`
- [x] `internal/cli/root.go` — root cmd, persistent `--config` flag
- [x] `internal/cli/serve.go` — `serve` cmd: load config, start whisper-server,
      start HTTP server, graceful shutdown on SIGINT/SIGTERM
- [ ] `serve.go` — warn at startup if ffmpeg binary not found (video uploads
      would fail)
- [x] `internal/cli/config.go` — `config init` (write example), `config get`,
      `config set`, `config path`
- [x] `internal/cli/version.go` — `version` (ldflags-injected)
- [x] wire flags to viper so `serve --port 9000` overrides config

## 6. Docs
- [ ] `README.md` — what it is, architecture diagram (ASCII), install,
      config reference (incl. audio section), CLI usage, OpenAI client example
      (curl + python), mp4/video note (ffmpeg dependency), security note
      (loopback-only whisper-server), build from source
- [x] `docs/research.md` — endpoint mapping table (split into
      research-openai.md + research-whisper-server.md)
- [ ] `.github/workflows/ci.yml` — go vet + test + build on push/PR

## 7. Verify
- [ ] `make build` compiles clean
- [ ] `make vet` clean
- [ ] `make test` passes (unit tests, no real GPU/ffmpeg needed)
- [ ] `gofmt -l .` empty
- [ ] manual smoke (if a whisper-server binary is available): start, hit
      `/v1/models`, POST a small wav to `/v1/audio/transcriptions`

## 8. Ship
- [ ] commit on branch `feat/whisper-oai`
- [ ] push to `git@github.com:tekmanic/whisper-oai.git`
      (use `github.pem` key, see TOOLS.md)
- [ ] open PR (optional)

---

## Key design decisions (locked)
1. **whisper-server is loopback-only.** It binds `127.0.0.1:<whisper.port>`;
   only the Go proxy can reach it. The public port is `server.port`.
2. **Go owns whisper-server lifecycle.** `serve` starts it as a child process,
   waits for `/health`, and tears it down on exit. No manual whisper-server.
3. **Cobra/Viper** for CLI + config. Precedence: flag > env > file > default.
4. **OpenAI-compatible subset.** We implement the audio transcription/translation
   + models endpoints (the ones that map to whisper). We do NOT implement
   chat/completions/images/etc.
5. **Model name** advertised as `whisper-1` (configurable via
   `server.model_name`), matching OpenAI's whisper model id.
6. **Video uploads (mp4 & friends).** whisper-server cannot decode video
   containers, so the proxy spools the upload to a temp file and, for video
   files, strips the audio track with ffmpeg into a 16 kHz mono WAV before
   forwarding. ffmpeg is a runtime dependency for video uploads only; plain
   audio (wav/mp3/ogg/flac/opus) needs no ffmpeg.

## Risks / open questions
- whisper-server flag names vary by version (`-dev` vs `--device`, `-t` vs
  `--threads`). The manager builds argv from config; verify against the
  installed binary's `--help` before relying on exact flags.
- `ggml-large-v3-turbo.bin` is a large download; not our job to fetch, but the
  README must document it.
- GPU device index `-dev 0` assumes CUDA. On CPU-only hosts the user sets
  `whisper.device` / `extra_args` accordingly.
- ffmpeg must be installed for video uploads; `serve` warns at startup if it
  is not on PATH (or at `audio.ffmpeg_bin`).
