# whisper-oai

A Golang wrapper around [whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s
`whisper-server` that exposes an **OpenAI-compatible** HTTP API. Point any
OpenAI client at it and your transcription calls run locally on your own
hardware — no API key, no per-token billing, no audio leaving the machine.

The Go app owns the whole lifecycle: it spawns `whisper-server` as a child
process, waits for it to be ready, proxies requests to it, and tears it down
on exit. `whisper-server` binds to **loopback only**, so the Go proxy is the
sole consumer of the backend port.

```
                        ┌────────────────────────────────────────────┐
  OpenAI client         │                whisper-oai (Go)            │
 ┌──────────┐  HTTPS    │  ┌──────────────────────────────────────┐  │  loopback
 │ curl /   │──────────▶│  │  OpenAI-compatible HTTP server       │  │  (127.0.0.1)
 │ python / │  /v1/...  │  │  :8000 (server.port)                 │  │
 │ openai   │◀──────────│  │  /v1/audio/transcriptions             │  │
 └──────────┘           │  │  /v1/audio/translations               │  │
                        │  │  /v1/models  /healthz                 │  │
                        │  └──────────────┬───────────────────────┘  │
                        │                 │ multipart                │
                        │  ┌──────────────▼───────────────────────┐  │
                        │  │  audio: spool upload → ffmpeg        │  │
                        │  │  (video → 16 kHz mono WAV)           │  │
                        │  └──────────────┬───────────────────────┘  │
                        │                 │ POST /inference          │
                        │  ┌──────────────▼───────────────────────┐  │
                        │  │  whisper-server (child process)      │  │
                        │  │  127.0.0.1:8080 (whisper.port)       │  │
                        │  │  ggml-large-v3-turbo, flash-attn     │  │
                        │  └──────────────────────────────────────┘  │
                        └────────────────────────────────────────────┘
```

## Features

- **OpenAI-compatible API** — `POST /v1/audio/transcriptions`,
  `POST /v1/audio/translations`, `GET /v1/models`, `GET /v1/models/{model}`,
  `GET /healthz`. Drop-in for the `openai` Python/Node SDKs (just change
  `base_url`).
- **Owns whisper-server** — started/stopped with the app; readiness-gated on
  `/health`; graceful SIGTERM → SIGKILL shutdown.
- **Video uploads (mp4 & friends)** — large video files are spooled to disk
  (not memory) and their audio track is stripped with **ffmpeg** into a
  16 kHz mono WAV before inference. Plain audio (wav/mp3/ogg/flac/opus)
  bypasses ffmpeg entirely.
- **Cobra/Viper CLI + config** — flags, env vars, and a YAML config file with
  clean precedence: `flag > env (WHISPER_OAI_*) > file > default`.
- **Optional auth** — set `server.api_key` and every request needs
  `Authorization: Bearer <key>`.
- **Loopback-only backend** — `whisper-server` is unreachable from the
  network; only the proxy can talk to it.

## Endpoints

| Method | Path                     | Notes                                            |
|--------|--------------------------|--------------------------------------------------|
| POST   | `/v1/audio/transcriptions` | multipart: `file`, `model`, `language`, `prompt`, `temperature`, `response_format` |
| POST   | `/v1/audio/translations`   | same as above (forces translation)               |
| GET    | `/v1/models`               | lists the configured model                       |
| GET    | `/v1/models/{model}`       | 404 unless it matches `server.model_name`        |
| GET    | `/healthz`                 | proxy liveness                                   |

`response_format` accepts `json` (default), `verbose_json`, `text`, `srt`,
`vtt`.

## Requirements

- **Go 1.22+** (to build whisper-oai)
- **whisper.cpp** with the `whisper-server` target built
- A **GGML model** (e.g. `ggml-large-v3-turbo.bin`)
- **ffmpeg** — only needed if you upload video files

### 1. Build whisper-server

```sh
git clone https://github.com/ggml-org/whisper.cpp /opt/whisper.cpp
cmake -B /opt/whisper.cpp/build -DGGML_CUDA=on   # -DGGML_CUDA=off for CPU
cmake --build /opt/whisper.cpp/build --config Release -j
# → /opt/whisper.cpp/build/bin/whisper-server
```

### 2. Download a model

```sh
mkdir -p /opt/whisper.cpp/models
cd /opt/whisper.cpp/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

### 3. Install ffmpeg (video uploads only)

```sh
# Debian/Ubuntu
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
```

### 4. Build whisper-oai

```sh
git clone git@github.com:tekmanic/whisper-oai.git
cd whisper-oai
make build          # → bin/whisper-oai
```

## Quick start

```sh
# 1. Create a config file from the example
./bin/whisper-oai config init

# 2. Start it (spawns whisper-server, waits for readiness, serves :8000)
./bin/whisper-oai serve

# or override on the command line (flags beat config file and env):
./bin/whisper-oai serve --port 9000 --whisper-model /opt/whisper.cpp/models/ggml-large-v3-turbo.bin
```

With the default config the app launches whisper-server exactly like:

```sh
/opt/whisper.cpp/build/bin/whisper-server \
  -m /opt/whisper.cpp/models/ggml-large-v3-turbo.bin \
  --flash-attn \
  -dev 0 \
  --host 127.0.0.1 --port 8080
```

### Try it

```sh
# List models
curl http://localhost:8000/v1/models

# Transcribe an audio file
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=whisper-1 \
  -F response_format=verbose_json

# Transcribe a VIDEO file (mp4) — audio is stripped automatically
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@presentation.mp4 \
  -F model=whisper-1

# Translate to English
curl http://localhost:8000/v1/audio/translations \
  -F file=@french_audio.mp3 \
  -F model=whisper-1
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# audio
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
print(result.text)

# video (mp4) works too — the server strips the audio track
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("presentation.mp4", "rb"),
)
print(result.text)
```

## CLI

```
whisper-oai serve     Start whisper-server and the OpenAI-compatible API
whisper-oai config    Manage the config file
  config init         Write the example config (refuses to overwrite without --force)
  config get <key>    Print one value (dotted key, e.g. server.port)
  config set <key> <value>
  config path         Print the resolved config file path
whisper-oai version   Print the version
```

`serve` flags (each maps to a config key): `--host`, `--port`, `--api-key`,
`--model-name`, `--whisper-bin`, `--whisper-model`, `--whisper-host`,
`--whisper-port`, `--device`, `--threads`, `--language`, `--no-flash-attn`.

## Configuration

Copy `config/whisper-oai.example.yaml` to `config/whisper-oai.yaml` (or point
`--config` elsewhere). Every key can also be set via env var
(`WHISPER_OAI_<SECTION>_<KEY>`, e.g. `WHISPER_OAI_SERVER_PORT`) or a flag.
Precedence: **flag > env > file > default**.

| Key | Default | Description |
|-----|---------|-------------|
| `server.host` | `0.0.0.0` | Interface the proxy listens on |
| `server.port` | `8000` | Public port (the "user's specified port") |
| `server.api_key` | `""` | If set, require `Authorization: Bearer <key>` |
| `server.model_name` | `whisper-1` | Model id advertised by `/v1/models` |
| `whisper.bin` | `/opt/whisper.cpp/build/bin/whisper-server` | whisper-server binary |
| `whisper.model` | `/opt/whisper.cpp/models/ggml-large-v3-turbo.bin` | GGML model file |
| `whisper.device` | `0` | GPU device index (`-dev`) |
| `whisper.flash_attn` | `true` | Enable flash attention (`--flash-attn`) |
| `whisper.host` | `127.0.0.1` | Backend bind host — **keep loopback** |
| `whisper.port` | `8080` | Backend port (internal) |
| `whisper.threads` | `0` | CPU threads (`-t`), 0 = auto |
| `whisper.language` | `auto` | Default language (`-l`) |
| `whisper.extra_args` | `[]` | Extra args appended verbatim |
| `audio.ffmpeg_bin` | `ffmpeg` | ffmpeg binary (video uploads only) |
| `audio.temp_dir` | `""` | Spool/extract dir (empty = `os.TempDir()`) |
| `audio.max_upload_mb` | `2048` | Max upload size in MB (0 = unlimited; 413 when exceeded) |
| `log.level` | `info` | `debug` \| `info` \| `warn` \| `error` |

## Video / large-file handling

whisper-server decodes raw audio (wav, mp3, ogg, flac, opus) but **not** video
containers. When you upload a video file (`.mp4`, `.mov`, `.mkv`, `.webm`,
`.avi`, `.wmv`, `.flv`, `.mpg`, `.mpeg`, `.3gp`, `.ts`, `.m2ts`, `.ogv`, or
any `video/*` content type), whisper-oai:

1. Spools the upload to a temp file so large files stream to disk instead of
   sitting in memory.
2. Runs ffmpeg to extract the audio track as a **16 kHz mono PCM WAV**
   (whisper.cpp's native input).
3. Forwards that WAV to whisper-server.

Plain audio uploads skip ffmpeg entirely. If ffmpeg is missing, `serve` logs a
warning at startup and video uploads return a 502 (audio uploads still work).

## Security notes

- **whisper-server is loopback-only.** It binds `127.0.0.1:<whisper.port>`;
  only the Go proxy can reach it. Do not change `whisper.host` to a public
  interface.
- **The public port is `server.port`.** If you expose it beyond localhost, set
  `server.api_key` and put TLS in front (reverse proxy).
- **Uploads are size-capped** by `audio.max_upload_mb` (HTTP 413 when
  exceeded) and spooled to a temp dir you control via `audio.temp_dir`.

## Development

```sh
make build      # build to bin/whisper-oai
make test       # unit tests (no GPU or real whisper-server needed)
make vet        # go vet
make fmt        # gofmt
make tidy       # go mod tidy
make run        # build + serve with the default config
```

Tests use `net/http/httptest` fakes for whisper-server and a scripted fake
ffmpeg, so the full suite runs on a plain CI box.

## Project layout

```
cmd/whisper-oai/        main entrypoint
internal/cli/           Cobra commands (serve, config, version)
internal/config/        Viper config loading + validation
internal/whisper/       whisper-server process manager + HTTP client
internal/server/        OpenAI-compatible HTTP server + handlers
internal/audio/         upload spooling + ffmpeg audio extraction
config/                 example config
docs/                   design + research notes
```

## License

MIT — see [LICENSE](LICENSE).
