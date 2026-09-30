# Research — OpenAI API (audio + models)

Source: `openai/openai-openapi` `openapi.yaml` (fetched 2026-09-29). Only the
endpoints whisper-oai mirrors are documented here. Field names are exact.

## Endpoints we implement

| OpenAI endpoint | whisper-server call | Notes |
|---|---|---|
| `POST /v1/audio/transcriptions` | `POST /inference` (`translate=false`) | multipart |
| `POST /v1/audio/translations` | `POST /inference` (`translate=true`) | multipart |
| `GET /v1/models` | (synthesized) | returns configured model |
| `GET /v1/models/{model}` | (synthesized) | returns configured model |

We do NOT implement chat/completions/images/speech/etc.

---

## `POST /v1/audio/transcriptions`

Multipart form fields (`CreateTranscriptionRequest`):

| field | type | required | notes |
|---|---|---|---|
| `file` | binary | **yes** | audio: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| `model` | string | **yes** | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, ... |
| `language` | string | no | ISO-639-1 (e.g. `en`). Improves accuracy/latency. |
| `prompt` | string | no | guide style / continue previous segment |
| `response_format` | string | no | `json` (default), `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | number | no | 0–1, default 0 |
| `timestamp_granularities[]` | string[] | no | `word`, `segment`; requires `verbose_json` |
| `include[]` | string[] | no | `logprobs` (gpt-4o models only) |
| `stream` | bool | no | NOT supported for `whisper-1` (ignored) |

(Other fields — `languages`, `keywords`, `chunking_strategy`,
`known_speaker_names`, `known_speaker_references` — are gpt-transcribe /
diarize only; we accept and ignore them.)

### Response — `response_format=json` (default)
```json
{ "text": "The transcribed text." }
```
(`usage` is optional; we omit it — whisper.cpp has no token accounting.)

### Response — `response_format=verbose_json`
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 3.32,
  "text": "The transcribed text.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.319999933242798,
      "text": " The transcribed text.",
      "tokens": [50364, 440, 7534],
      "temperature": 0.0,
      "avg_logprob": -0.2860786020755768,
      "compression_ratio": 1.2363636493682861,
      "no_speech_prob": 0.00985979475080967
    }
  ]
}
```
`segments[]` fields: `id`, `seek`, `start`, `end`, `text`, `tokens[]`,
`temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob`.
`words[]` (word-level timestamps) is supported when
`timestamp_granularities` includes `word`; each word is
`{ "word": string, "start": number, "end": number }`.

### Response — `text` / `srt` / `vtt`
Plain body with `Content-Type: text/plain` (text), or the subtitle format.
whisper-server produces `srt` and `vtt` natively.

---

## `POST /v1/audio/translations`

Multipart form fields (`CreateTranslationRequest`):

| field | type | required | notes |
|---|---|---|---|
| `file` | binary | **yes** | same audio formats |
| `model` | string | **yes** | `whisper-1` |
| `prompt` | string | no | should be in English |
| `response_format` | string | no | `json` (default), `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | number | no | 0–1, default 0 |

Note: **no `language` field** on translations (output is always English).

### Response — `json`
```json
{ "text": "The translated text." }
```

### Response — `verbose_json`
```json
{
  "task": "translate",
  "language": "english",
  "duration": 3.32,
  "text": "The translated text.",
  "segments": [ /* same TranscriptionSegment shape */ ]
}
```

---

## `GET /v1/models`
```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1687882411,
      "owned_by": "whisper-oai"
    }
  ]
}
```

## `GET /v1/models/{model}`
```json
{
  "id": "whisper-1",
  "object": "model",
  "created": 1687882411,
  "owned_by": "whisper-oai"
}
```
`404` if the model id doesn't match the configured `server.model_name`.

---

## Error shape (all endpoints)
```json
{
  "error": {
    "message": "Invalid value for 'model'.",
    "type": "invalid_request_error",
    "param": "model",
    "code": null
  }
}
```
`type` values we use: `invalid_request_error` (400), `authentication_error`
(401), `server_error` (500), `service_unavailable` (503).

---

## Mapping to whisper-server (see research-whisper-server.md)

| OpenAI `response_format` | whisper `response_format` | post-processing |
|---|---|---|
| `json` | `json` | return `{text}` (whisper json is `{"text":...}`) |
| `verbose_json` | `verbose_json` | map whisper segments → OpenAI segment shape |
| `text` | `text` | passthrough |
| `srt` | `srt` | passthrough |
| `vtt` | `vtt` | passthrough |

- OpenAI `language` → whisper `language`
- OpenAI `prompt` → whisper `prompt`
- OpenAI `temperature` → whisper `temperature`
- transcriptions → whisper `translate=false`; translations → `translate=true`
- whisper `verbose_json` segments already carry `id`, `start`, `end`, `text`,
  `tokens`, `temperature`, `avg_logprob`, `no_speech_prob`. We add `seek`
  (0) and `compression_ratio` (0.0) to match the OpenAI shape.
