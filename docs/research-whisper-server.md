# whisper.cpp `whisper-server` — HTTP API & CLI Reference

> Research date: 2026-09-29
> Ground truth: `examples/server/server.cpp` on `master` of `github.com/ggml-org/whisper.cpp`
> (fetched 2026-09-29), plus `examples/server/README.md` and the root `README.md`.
> Where the README and the code disagree, **the code wins** (discrepancies are flagged in §9).
>
> Purpose: reference for a Go wrapper that (a) spawns `whisper-server` with the right argv
> and (b) calls its HTTP API. No Go code here — exact names only.

---

## 1. Endpoints

All paths are prefixed by `--request-path` (default: empty string). The inference path is
configurable via `--inference-path` (default: `/inference`).

| Method | Path (default) | Purpose |
|---|---|---|
| `POST` | `/inference` | Transcribe an uploaded audio file |
| `POST` | `/load` | Hot-swap the loaded model (path on the server's filesystem) |
| `GET` | `/health` | Readiness probe |
| `GET` | `/` | Built-in HTML index page (only served if no `index.html` exists in `--public` dir) |
| `OPTIONS` | `/inference` | CORS preflight (empty 200 body; CORS headers always present) |
| any | static files under `--public` dir (default `examples/server/public`) | Static file serving via `set_base_dir` |

**Endpoints that do NOT exist** (confirmed absent from `server.cpp`):
- `GET /inference` — no GET handler on the inference path.
- `GET /models` — no such endpoint. Model listing is not exposed; only `POST /load` exists.
- `GET /timestamps` — no such endpoint.
- `GET /speakers` — no such endpoint.

Unknown paths fall into the error handler: 404 with body `File Not Found (<path>)` (text/plain).

---

## 2. `POST /inference` — request

`Content-Type: multipart/form-data`. The **only required field is `file`**; every other field
is optional and, when absent, falls back to the value baked in from the CLI at startup
(`whisper_params default_params` is snapshotted at startup and reset per request).

### 2.1 Field table (exact names, from `get_req_parameters()`)

| Field | Parsed as | Notes |
|---|---|---|
| `file` | file upload (required) | Audio. Decoded in-memory by miniaudio — **16-bit PCM WAV is the supported format** unless `--convert` is set (see §2.4). Missing → 400 `{"error":"no 'file' field in the request"}`. |
| `temperature` | float (`std::stof`) | Default 0.0. Fallback behavior: see §2.3. |
| `temperature_inc` | float | Default 0.2. |
| `response_format` | string | **This is the field name — not `response`.** Values: `json` (default), `text`, `srt`, `vtt`, `verbose_json`. **There is no `tsv` and no `nfo`.** Unknown values fall through to the `json` branch (the `else` clause), so a typo silently gives you `json`. |
| `language` | string | Default `en`. `auto` = auto-detect. Not validated per-request (validated only at startup for the CLI value). |
| `prompt` | string | Initial prompt. Default empty. |
| `carry_initial_prompt` | bool string | Default false. |
| `translate` | bool string | Default false. |
| `detect_language` | bool string | Default false. If true, forces `language=auto` and adds `language` to the plain `json` response. |
| `no_timestamps` | bool string | Default false. |
| `token_timestamps` | bool string | Default: computed — see §2.2. |
| `no_speech_thold` | float | Default 0.6. (Note the typo "thold" — it is in the upstream code.) |
| `word_thold` | float | Default 0.01. |
| `entropy_thold` | float | Default 2.40. |
| `logprob_thold` | float | Default -1.00. |
| `max_context` | int | Default -1 (unlimited). |
| `max_len` | int | Default 0 → internally becomes 60 (max segment length in chars). |
| `best_of` | int | Default 2. |
| `beam_size` | int | Default -1. `> 1` switches sampling strategy from greedy to beam search. |
| `audio_ctx` | float (parsed with `std::stof`!) | Default 0 (all). |
| `offset_t` | int | Time offset in ms. Default 0. |
| `offset_n` | int | Segment index offset (used in SRT numbering). Default 0. |
| `duration` | int | Duration to process in ms. Default 0 (all). |
| `split_on_word` | bool string | Default false. |
| `debug_mode` | bool string | Default false. |
| `diarize` | bool string | Default false. Stereo energy-based speaker tagging. |
| `tinydiarize` | bool string | Default false. Requires a tdrz model. |
| `suppress_nst` | bool string | Default false. Suppress non-speech tokens. |
| `suppress_non_speech` | bool string | Alias of `suppress_nst` (both are checked; `suppress_nst` wins if both present, since it's parsed last). |
| `no_language_probabilities` | bool string | Default false. If true, omits the expensive `detected_language*` / `language_probabilities` fields from `verbose_json`. |
| `vad` | bool string | Default false. |
| `vad_threshold` | float | Default 0.5. |
| `vad_min_speech_duration_ms` | float (parsed with `std::stof`!) | Default 250. |
| `vad_min_silence_duration_ms` | float (parsed with `std::stof`!) | Default 100. |
| `vad_max_speech_duration_s` | float | Default FLT_MAX. |
| `vad_speech_pad_ms` | int | Default 30. |
| `vad_samples_overlap` | float | Default 0.1. |

**Bool parsing** (`parse_str_to_bool`): a field is true **only** if its value is exactly
`true`, `1`, `yes`, or `y` (case-sensitive). Anything else (`True`, `TRUE`, `0`, `false`,
garbage) is false. Send lowercase `true`/`false`.

**Fields that do NOT exist** (confirmed absent):
- `n_threads` / `threads` — **thread count is CLI-only** (`-t`). It cannot be set per request.
- `max_tokens` — no such field.
- `model` — not accepted on `/inference` (only on `/load`).

### 2.2 `token_timestamps` default logic

If the request does **not** include `token_timestamps`, the server computes:

```
token_timestamps = !no_timestamps && (response_format == "verbose_json" || max_len > 0 || split_on_word)
```

So with the defaults (`response_format=json`, `max_len=0`, `split_on_word=false`) it is **off**.
It is on by default for `verbose_json`.

### 2.3 Temperature fallback behavior

The server passes `temperature`, `temperature_inc`, `logprob_thold`, `entropy_thold` straight
into `whisper_full()`. The fallback loop lives in the whisper.cpp core (`whisper_full`), not
in the server: it decodes at `temperature`; if the result's `avg_logprob < logprob_thold` or
`compression_ratio > entropy_thold`, it retries at `temperature + temperature_inc`, up to 8
attempts. Defaults: `temperature=0.0`, `temperature_inc=0.2`, `logprob_thold=-1.0`,
`entropy_thold=2.4`.

Note: the CLI flag `-nf/--no-fallback` is parsed by the server binary but **never wired into
the per-request `whisper_full_params`** — temperature fallback is always active in the server.

### 2.4 Audio decoding

- Without `--convert`: the uploaded bytes are decoded in memory by miniaudio. Expect
  **16-bit PCM WAV** (other formats may fail → 400 `{"error":"failed to read audio data"}`).
- With `--convert` (requires `ffmpeg` on the server's PATH): the upload is written to a temp
  file (`--tmp-dir`, default `.`) and converted via
  `ffmpeg -i <tmp> -y -ar 16000 -ac <1|2> -c:a pcm_s16le <tmp>_temp.wav` (stereo only if
  `diarize=true`), then read back. Conversion failure → 500 with one of:
  `{"error":"FFmpeg conversion failed."}`, `{"error":"Failed to execute ffmpeg command."}`,
  `{"error":"Failed to remove the original file."}`, `{"error":"Failed to rename the temporary file."}`.
  At startup with `--convert`, the server runs `ffmpeg -version` and **exits(0)** if ffmpeg is missing.

### 2.5 Other per-request behavior

- If the loaded model is **not multilingual** and the request sets `language != "en"` or
  `translate=true`, the server logs a warning and silently forces `language=en`, `translate=false`.
- `detect_language=true` forces `language=auto`.
- Inference is aborted if the HTTP client disconnects (abort callback checks
  `is_connection_closed()`); the server then returns 499 `{"error":"client disconnected"}`.
- A single global mutex serializes all `/inference` and `/load` requests: **one inference at a
  time**, regardless of HTTP concurrency.

---

## 3. Response formats

### 3.1 `response_format=json` (default)

Content-Type: `application/json`.

```json
{
  "text": "And so my fellow Americans, ask not what your country can do for you..."
}
```

- `text` (string): all segment texts joined with `\n` (each segment's text as-is, with the
  leading space whisper produces; diarize prefix `(speaker N)` prepended when diarizing stereo).
- `language` (string, e.g. `"english"`): **present only if `detect_language=true`**.

That's it. The plain `json` format has **no segments array**.

### 3.2 `response_format=verbose_json`

Content-Type: `application/json`. Modeled on OpenAI whisper's Python output.

Top level:

| Field | Type | Notes |
|---|---|---|
| `task` | string | `"transcribe"` or `"translate"` |
| `language` | string | Full language name, e.g. `"english"` (from `whisper_lang_str_full`) |
| `duration` | float | Audio duration in seconds (`n_samples / 16000`) |
| `text` | string | Same as plain `json` |
| `detected_language` | string | Full name. **Only if `no_language_probabilities` is false** (default: included). |
| `detected_language_probability` | float | Probability of the detected language. Same condition. |
| `language_probabilities` | object | Map of language code → float. Same condition. Only entries with prob > 0.001. |
| `segments` | array | See below. |

Each segment object (field order as emitted):

| Field | Type | Notes |
|---|---|---|
| `id` | int | 0-based segment index |
| `text` | string | Segment text |
| `start` | float | Seconds (`t0 * 0.01`). **Omitted if `no_timestamps=true`** |
| `end` | float | Seconds (`t1 * 0.01`). **Omitted if `no_timestamps=true`** |
| `speaker` | string | `"0"`, `"1"`, or `"?"`. **Only if `diarize=true` and input is stereo** |
| `tokens` | array of int | Token IDs (EOT tokens excluded; multi-byte UTF-8 continuation tokens are merged into the preceding word) |
| `words` | array of objects | See below |
| `temperature` | float | The request's temperature (the base value, not the fallback-adjusted one) |
| `avg_logprob` | float | `total_logprob / n_tokens` for the segment |
| `no_speech_prob` | float | From `whisper_full_get_segment_no_speech_prob` |

**`compression_ratio` is NOT emitted** — there is a `// TODO compression_ratio and no_speech_prob
are not implemented yet` comment in the code with the line commented out. Do not parse it.

Each word object:

| Field | Type | Notes |
|---|---|---|
| `word` | string | Word text (may include leading space) |
| `start` | float | Seconds. **Only if `!no_timestamps && token_timestamps`** |
| `end` | float | Seconds. Same condition. |
| `t_dtw` | int | DTW timestamp (only meaningful with `--dtw`). Same condition. |
| `probability` | float | Token probability |

### 3.3 `response_format=text`

Content-Type: **`text/html; charset=utf-8`** (yes, html — upstream quirk). Body: segment texts
joined with `\n`, no trailing structure.

### 3.4 `response_format=srt`

Content-Type: `application/x-subrip`. Standard SRT:

```
1
00:00:00,000 --> 00:00:01,590
  And so my fellow Americans...

2
...
```

- Index = `i + 1 + offset_n`.
- Timestamps use `to_timestamp(t, true)` → `HH:MM:SS,mmm`.
- Diarize prefix `(speaker N)` prepended to the line when diarizing stereo.

### 3.5 `response_format=vtt`

Content-Type: `text/vtt`.

```
WEBVTT

00:00:00.000 --> 00:00:01.590
  And so my fellow Americans...

```

- Timestamps use `to_timestamp(t)` → `HH:MM:SS.mmm` (dot, not comma).
- Diarize renders as `<v SpeakerN>` cue identifier.

### 3.6 Error responses (all `application/json` unless noted)

| Status | Body | When |
|---|---|---|
| 400 | `{"error":"no 'file' field in the request"}` | `file` missing |
| 400 | `{"error":"failed to read audio data"}` | miniaudio decode failure |
| 400 | `{"error":"failed to read WAV file"}` | decode failure on `--convert` path |
| 400 | `{"error":"no 'model' field in the request"}` | `/load` without `model` |
| 400 | `{"error":"model not found!"}` | `/load` path doesn't exist on server FS |
| 499 | `{"error":"client disconnected"}` | client closed connection mid-inference |
| 500 | `{"error":"failed to process audio"}` | `whisper_full_parallel` failed |
| 500 | `{"error":"FFmpeg conversion failed."}` etc. | `--convert` path failures (see §2.4) |
| 500 | `500 Internal Server Error\n<exception>` (text/plain) | uncaught C++ exception |
| 404 | `File Not Found (<path>)` (text/plain) | unknown path |
| 400 | `Invalid request` (text/plain) | malformed request via error handler |

---

## 4. `GET /health`

| State | Status | Body |
|---|---|---|
| Model loaded & ready | 200 | `{"status":"ok"}` |
| Model loading (startup, or after a `/load`) | 503 | `{"status":"loading model"}` |

Use 200 as the readiness gate before sending `/inference` traffic.

---

## 5. `POST /load`

Multipart form, single field:

| Field | Type | Notes |
|---|---|---|
| `model` | **string** (form field, not a file upload) | Absolute/relative path to a ggml model file **on the whisper-server's filesystem**. The content is used as a path string (`req.get_file_value("model").content`), then checked with `is_file_exist`. |

Behavior:
- Sets state to loading (`/health` → 503) while swapping.
- Frees the old context, loads the new one with the **same** context params (GPU/flash-attn/device from startup).
- Success: 200, body `Load was successful!`, Content-Type `application/text`.
- If the new model fails to initialize, the process **exits(1)** — the server does not survive a bad `/load`.

---

## 6. CLI flags of `whisper-server` (exact, from `whisper_params_parse`)

### 6.1 Flags relevant to spawning (short + long)

| Short | Long | Arg | Default | Meaning |
|---|---|---|---|---|
| `-m` | `--model` | FNAME | `models/ggml-base.en.bin` (relative to CWD) | Model path. **Required in practice** — default is a repo-relative path that won't exist on a deployed host. |
| `-t` | `--threads` | N | `min(4, hardware_concurrency)` | CPU threads per processor |
| `-p` | `--processors` | N | 1 | Parallel processors (`whisper_full_parallel`) |
| `-l` | `--language` | LANG | `en` | Default language (`auto` = detect). Unknown non-`auto` value at startup → exit. |
| `—` | `--prompt` | PROMPT | `""` | Default initial prompt (long-only) |
| `-tr` | `--translate` | — | false | Default task = translate |
| `-ng` | `--no-gpu` | — | (gpu on) | **Disable GPU entirely** |
| `-dev` | `--device` | N | 0 | GPU device ID (long+short both exist) |
| `-fa` | `--flash-attn` | — | **true** (code default) | Enable flash attention |
| `-nfa` | `--no-flash-attn` | — | — | Disable flash attention |
| `-nt` | `--no-timestamps` | — | false | Default: no timestamps |
| `-nf` | `--no-fallback` | — | false | Parsed but **not wired into server inference** (no-op in practice) |
| `-nlp` | `--no-language-probabilities` | — | false | Omit language probabilities from `verbose_json` |
| `-sns` | `--suppress-nst` | — | false | Suppress non-speech tokens |
| `-nth` | `--no-speech-thold` | N | 0.60 | No-speech threshold |
| `-bo` | `--best-of` | N | 2 | Greedy best-of |
| `-bs` | `--beam-size` | N | -1 | Beam size (>1 → beam search) |
| `-ml` | `--max-len` | N | 0 | Max segment length in chars (0 → 60) |
| `-mc` | `--max-context` | N | -1 | Max text context tokens |
| `-wt` | `--word-thold` | N | 0.01 | Word timestamp prob threshold |
| `-et` | `--entropy-thold` | N | 2.40 | Entropy threshold for decoder fail |
| `-lpt` | `--logprob-thold` | N | -1.00 | Logprob threshold for decoder fail |
| `-ot` | `--offset-t` | N | 0 | Time offset ms |
| `-on` | `--offset-n` | N | 0 | Segment index offset |
| `-d` | `--duration` | N | 0 | Process duration ms |
| `-ac` | `--audio-ctx` | N | 0 | Audio context size |
| `-sow` | `--split-on-word` | — | false | Split on word not token |
| `-di` | `--diarize` | — | false | Stereo diarization (mutually exclusive with `-tdrz`) |
| `-tdrz` | `--tinydiarize` | — | false | tinydiarize (needs tdrz model) |
| `-dl` | `--detect-language` | — | false | Detect language |
| `-debug` | `--debug-mode` | — | false | Debug mode |
| `-ps` | `--print-special` | — | false | Print special tokens |
| `-pc` | `--print-colors` | — | false | ANSI colors in stdout |
| `-pr` | `--print-realtime` | — | false | Realtime segment printing |
| `-pp` | `--print-progress` | — | false | Progress printing |
| `-fp` | `--font-path` | PATH | macOS Courier path | Parsed, not in usage text |
| `-oved` | `--ov-e-device` | DNAME | `CPU` | OpenVINO encode device |
| `-dtw` | `--dtw` | MODEL | `""` | DTW token timestamps (presets: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large.v1`, `large.v2`, `large.v3`, `large.v3.turbo`) |
| `-h` | `--help` | — | — | Print usage, exit 0 |

Server-specific (long-only):

| Long | Arg | Default | Meaning |
|---|---|---|---|
| `--host` | HOST | `127.0.0.1` | Bind hostname/IP. Use `0.0.0.0` to expose. |
| `--port` | PORT | `8080` | Bind port |
| `--public` | PATH | `examples/server/public` | Static file dir |
| `--request-path` | PATH | `""` | Prefix for all routes |
| `--inference-path` | PATH | `/inference` | Inference route suffix |
| `--convert` | — | false | ffmpeg-transcode uploads to 16 kHz WAV (needs ffmpeg on PATH) |
| `--tmp-dir` | PATH | `.` | Temp dir for `--convert` |

VAD (long + short): `--vad`, `-vm/--vad-model`, `-vt/--vad-threshold`,
`-vspd/--vad-min-speech-duration-ms`, `-vsd/--vad-min-silence-duration-ms`,
`-vmsd/--vad-max-speech-duration-s`, `-vp/--vad-speech-pad-ms`, `-vo/--vad-samples-overlap`.

Environment variable: **`WHISPER_ARG_DEVICE`** (read before argv) sets the GPU device ID.

Unknown argument → prints usage and **exits with code 0** (not 1) — check for process exit
plus stderr, not just exit code, when validating argv.

### 6.2 Example argv (CPU-only, what a Go wrapper would spawn)

```
whisper-server \
  -m /path/to/models/ggml-base.en.bin \
  -t 8 \
  -l en \
  -ng \
  -nfa \
  --host 127.0.0.1 \
  --port 8080
```

GPU variant: drop `-ng`, keep `-fa` (default on), optionally `-dev 1`.

---

## 7. Building

From the repo root (root README, "Quick start"):

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

cmake -B build
cmake --build build -j --config Release
```

- Binary lands at **`build/bin/whisper-server`** (all example binaries go to `build/bin/`).
- GPU backends (pick one at configure time):
  - NVIDIA CUDA: `cmake -B build -DGGML_CUDA=1`
  - Vulkan: `cmake -B build -DGGML_VULKAN=1`
  - AMD ROCm: `cmake -B build -DGGML_HIP=1 -DAMDGPU_TARGETS="gfx1100"`
  - OpenBLAS CPU: `cmake -B build -DGGML_BLAS=1`
  - FFmpeg decode support (examples): `cmake -B build -DWHISPER_COMMON_FFMPEG=yes`
- Model download: `sh ./models/download-ggml-model.sh base.en` → `models/ggml-base.en.bin`.
- Docker images exist (`ghcr.io/ggml-org/whisper.cpp:main`, `:main-cuda`, `:main-vulkan`, …)
  and run `whisper-server --host 0.0.0.0 -m /models/ggml-base.bin`.

---

## 8. CORS & auth

- **CORS: present, wide open.** Every response carries default headers:
  - `Server: whisper.cpp`
  - `Access-Control-Allow-Origin: *`
  - `Access-Control-Allow-Headers: content-type, authorization`
  - Plus an `OPTIONS /inference` handler (empty 200) for preflight.
- **Auth: none.** No API key, no token, no TLS. The upstream README explicitly warns:
  "Do not run the server example with administrative privileges and ensure it's operated in a
  sandbox environment, especially since it involves risky operations like accepting user file
  uploads and using ffmpeg for format conversions."
- Bind to `127.0.0.1` (the default) and put your own auth/TLS in front if it must be reachable
  beyond localhost.

---

## 9. Gotchas & README/code discrepancies

1. **`response_format`, not `response`.** The multipart field is `response_format`.
2. **No `tsv`/`nfo` formats.** Values are exactly: `json`, `text`, `srt`, `vtt`, `verbose_json`.
3. **No per-request thread control.** `-t` is startup-only.
4. **`compression_ratio` is not in `verbose_json`** (TODO in code). `no_speech_prob` IS present.
5. **README (examples/server/README.md) is stale** vs. current `server.cpp`:
   - README shows `-fa, --flash-attn [false]` default; **code default is `true`**.
   - README lists `-nc, --no-context` — **not parsed by the server code** (`no_context` is
     hardcoded `true` in the params struct and always passed to `whisper_full`).
   - README omits `-dev`, `-nfa`, `-nlp`, `--carry-initial-prompt`, `--tmp-dir`, `-fp`,
     `--request-path`/`--inference-path` details.
   - README's usage block omits `--inference-path` default display and several VAD flags.
6. **`-nf/--no-fallback` is a no-op in the server** (parsed, never applied to `whisper_full_params`).
7. **`/load` takes a filesystem path string**, not a model upload — the Go wrapper must ensure
   the model file is visible to the whisper-server process (same host/container).
8. **Bad `/load` kills the process** (`exit(1)`) — the supervisor must restart it.
9. **Single-flight inference**: one global mutex; concurrent `/inference` requests queue.
10. **Timeouts**: read and write timeouts are 600 s each (hardcoded, not configurable).
11. **`text` format returns Content-Type `text/html; charset=utf-8`** — don't gate on content type.
12. **Bool fields are case-sensitive**: only `true`/`1`/`yes`/`y` are truthy.
13. **`audio_ctx`, `vad_min_speech_duration_ms`, `vad_min_silence_duration_ms` are parsed as
    floats** (`std::stof`) even though they're conceptually ints — send plain integers, they'll
    parse fine.
14. **Startup language validation**: `-l <unknown>` (other than `auto`) exits the process at
    startup. Per-request `language` is not validated.
15. **Unknown CLI args exit with code 0** (usage printed) — don't rely on non-zero exit for
    argv validation.
16. **`GET /`** serves a built-in HTML page (with a curl example and a browser form) only when
    no `index.html` exists in the `--public` directory.

---

## 10. Minimal curl examples (from upstream)

```bash
# transcribe (json)
curl 127.0.0.1:8080/inference \
  -H "Content-Type: multipart/form-data" \
  -F file="@jfk.wav" \
  -F temperature="0.0" \
  -F temperature_inc="0.2" \
  -F prompt="<prompt>" \
  -F carry_initial_prompt="true" \
  -F response_format="json"

# verbose_json with word timestamps
curl 127.0.0.1:8080/inference \
  -F file="@jfk.wav" \
  -F response_format="verbose_json" \
  -F language="auto"

# health
curl 127.0.0.1:8080/health

# hot-swap model
curl 127.0.0.1:8080/load \
  -H "Content-Type: multipart/form-data" \
  -F model="/path/to/ggml-small.en.bin"
```
