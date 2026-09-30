package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/tekmanic/whisper-oai/internal/config"
	"github.com/tekmanic/whisper-oai/internal/whisper"
)

// Canned whisper-server response bodies (upstream shapes).
const (
	cannedText = "And so my fellow Americans, ask not what your country can do for you."

	cannedJSON = `{"text":"And so my fellow Americans, ask not what your country can do for you."}`

	// Upstream verbose_json omits seek and compression_ratio (whisper.cpp
	// TODO); the client is expected to zero them.
	cannedVerboseJSON = `{"task":"transcribe","language":"english","duration":3.32,"text":"And so my fellow Americans, ask not what your country can do for you.","segments":[{"id":0,"text":" And so my fellow Americans, ask not what your country can do for you.","start":0.0,"end":3.32,"tokens":[50364,440,7534],"temperature":0.0,"avg_logprob":-0.2860786020755768,"no_speech_prob":0.00985979475080967}]}`

	cannedSRT = "1\n00:00:00,000 --> 00:00:03,320\n And so my fellow Americans, ask not what your country can do for you.\n"

	cannedVTT = "WEBVTT\n\n00:00:00.000 --> 00:00:03.320\n And so my fellow Americans, ask not what your country can do for you.\n"
)

// fakeWhisper is a canned whisper-server for tests. It records the last
// /inference request so tests can assert the OpenAI→whisper field mapping.
type fakeWhisper struct {
	mu       sync.Mutex
	fields   map[string]string
	filename string
	fileBody []byte
	fail     bool
}

// snapshot returns a copy of the last recorded request.
func (f *fakeWhisper) snapshot() (map[string]string, string, []byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fields := make(map[string]string, len(f.fields))
	for k, v := range f.fields {
		fields[k] = v
	}
	return fields, f.filename, f.fileBody
}

func (f *fakeWhisper) setFail(fail bool) {
	f.mu.Lock()
	f.fail = fail
	f.mu.Unlock()
}

func (f *fakeWhisper) serveInfer(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(1 << 20); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad multipart"}`))
		return
	}
	fields := map[string]string{}
	for k, v := range r.MultipartForm.Value {
		fields[k] = v[len(v)-1]
	}
	var filename string
	var body []byte
	if files := r.MultipartForm.File["file"]; len(files) > 0 {
		filename = files[0].Filename
		if fh, err := files[0].Open(); err == nil {
			body, _ = io.ReadAll(fh)
			_ = fh.Close()
		}
	}
	f.mu.Lock()
	f.fields, f.filename, f.fileBody = fields, filename, body
	fail := f.fail
	f.mu.Unlock()

	if fail {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"failed to process audio"}`))
		return
	}
	switch fields["response_format"] {
	case "verbose_json":
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(cannedVerboseJSON))
	case "text":
		// Upstream quirk: text format is served as text/html.
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(cannedText))
	case "srt":
		w.Header().Set("Content-Type", "application/x-subrip")
		_, _ = w.Write([]byte(cannedSRT))
	case "vtt":
		w.Header().Set("Content-Type", "text/vtt")
		_, _ = w.Write([]byte(cannedVTT))
	default: // json
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(cannedJSON))
	}
}

// newFakeWhisper starts a fake whisper-server (health + inference).
func newFakeWhisper(t *testing.T) (*httptest.Server, *fakeWhisper) {
	t.Helper()
	f := &fakeWhisper{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		case r.Method == http.MethodPost && r.URL.Path == "/inference":
			f.serveInfer(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, f
}

// testConfig returns a fresh config with the default model name.
func testConfig() *config.Config {
	return &config.Config{
		Server: config.ServerConfig{
			Host:      "127.0.0.1",
			Port:      8000,
			APIKey:    "",
			ModelName: "whisper-1",
		},
		Whisper: config.WhisperConfig{
			Host: "127.0.0.1",
			Port: 8080,
		},
	}
}

// newTestServer wires a real *Server whose whisper.Client points at the fake
// whisper-server. The Manager is constructed but never started (no process).
func newTestServer(t *testing.T, cfg *config.Config, fakeURL string) *Server {
	t.Helper()
	host, portStr, err := net.SplitHostPort(strings.TrimPrefix(fakeURL, "http://"))
	if err != nil {
		t.Fatalf("parse fake URL %q: %v", fakeURL, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("parse fake port %q: %v", portStr, err)
	}
	cfg.Whisper.Host = host
	cfg.Whisper.Port = port
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := whisper.NewManager(cfg.Whisper, log)
	return New(cfg, mgr, log)
}

// multipartRequest builds a POST multipart request. An empty filename omits
// the file field entirely.
func multipartRequest(t *testing.T, path string, fields map[string]string, filename string, content []byte) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatalf("write field %s: %v", k, err)
		}
	}
	if filename != "" {
		fw, err := mw.CreateFormFile("file", filename)
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		if _, err := fw.Write(content); err != nil {
			t.Fatalf("write file content: %v", err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func TestHealthz(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body["status"] != "ok" {
		t.Fatalf("status = %q, want ok", body["status"])
	}
}

func TestModels(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var body struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body.Object != "list" {
		t.Errorf("object = %q, want list", body.Object)
	}
	if len(body.Data) != 1 || body.Data[0].ID != "whisper-1" {
		t.Fatalf("data = %+v, want a single whisper-1 model", body.Data)
	}
	if body.Data[0].Object != "model" || body.Data[0].OwnedBy != "whisper-oai" {
		t.Errorf("data[0] = %+v, want object=model owned_by=whisper-oai", body.Data[0])
	}
}

func TestModel(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	t.Run("configured model", func(t *testing.T) {
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models/whisper-1", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
		}
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if body["id"] != "whisper-1" || body["object"] != "model" || body["owned_by"] != "whisper-oai" {
			t.Errorf("body = %v", body)
		}
	})

	t.Run("unknown model", func(t *testing.T) {
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models/gpt-4o-transcribe", nil))
		if rec.Code != http.StatusNotFound {
			t.Fatalf("status = %d, want 404 (body %s)", rec.Code, rec.Body.String())
		}
		var body struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Param   string `json:"param"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if body.Error.Type != "invalid_request_error" || body.Error.Param != "model" {
			t.Errorf("error = %+v, want invalid_request_error with param=model", body.Error)
		}
	})
}

func TestAuthMiddleware(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("ok"))
	})

	t.Run("empty key is passthrough", func(t *testing.T) {
		rec := httptest.NewRecorder()
		authMiddleware("", next).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
	})

	t.Run("missing header is 401", func(t *testing.T) {
		rec := httptest.NewRecorder()
		authMiddleware("secret", next).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401 (body %s)", rec.Code, rec.Body.String())
		}
		var body struct {
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if body.Error.Type != "authentication_error" {
			t.Errorf("error type = %q, want authentication_error", body.Error.Type)
		}
	})

	t.Run("wrong key is 401", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("Authorization", "Bearer wrong")
		rec := httptest.NewRecorder()
		authMiddleware("secret", next).ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", rec.Code)
		}
	})

	t.Run("correct key passes", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		authMiddleware("secret", next).ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		if rec.Body.String() != "ok" {
			t.Errorf("body = %q, want ok", rec.Body.String())
		}
	})
}

func TestAuthEndToEnd(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	cfg := testConfig()
	cfg.Server.APIKey = "secret"
	s := newTestServer(t, cfg, fake.URL)

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("no key: status = %d, want 401", rec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("with key: status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
}

func TestTranscriptionJSON(t *testing.T) {
	fake, state := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
		"model":           "whisper-1",
		"response_format": "json",
		"language":        "en",
		"prompt":          "fellow Americans",
		"temperature":     "0.5",
	}, "audio.wav", []byte("RIFF-fake-wav-bytes"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type = %q, want application/json", ct)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body["text"] != cannedText {
		t.Errorf("text = %v, want %q", body["text"], cannedText)
	}

	// Assert the OpenAI→whisper field mapping.
	fields, filename, fileBody := state.snapshot()
	if fields["translate"] != "false" {
		t.Errorf("translate = %q, want false", fields["translate"])
	}
	if fields["language"] != "en" {
		t.Errorf("language = %q, want en", fields["language"])
	}
	if fields["prompt"] != "fellow Americans" {
		t.Errorf("prompt = %q, want %q", fields["prompt"], "fellow Americans")
	}
	if got, _ := strconv.ParseFloat(fields["temperature"], 64); got != 0.5 {
		t.Errorf("temperature = %q, want 0.5", fields["temperature"])
	}
	if filename != "audio.wav" {
		t.Errorf("filename = %q, want audio.wav", filename)
	}
	if string(fileBody) != "RIFF-fake-wav-bytes" {
		t.Errorf("file body = %q, want RIFF-fake-wav-bytes", fileBody)
	}
}

func TestTranscriptionVerboseJSON(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
		"model":           "whisper-1",
		"response_format": "verbose_json",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var body struct {
		Task     string  `json:"task"`
		Language string  `json:"language"`
		Duration float64 `json:"duration"`
		Text     string  `json:"text"`
		Segments []struct {
			ID               int     `json:"id"`
			Seek             int     `json:"seek"`
			Start            float64 `json:"start"`
			End              float64 `json:"end"`
			Text             string  `json:"text"`
			Tokens           []int64 `json:"tokens"`
			Temperature      float64 `json:"temperature"`
			AvgLogprob       float64 `json:"avg_logprob"`
			CompressionRatio float64 `json:"compression_ratio"`
			NoSpeechProb     float64 `json:"no_speech_prob"`
		} `json:"segments"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v (body %s)", err, rec.Body.String())
	}
	if body.Task != "transcribe" {
		t.Errorf("task = %q, want transcribe", body.Task)
	}
	if body.Language != "english" {
		t.Errorf("language = %q, want english", body.Language)
	}
	if body.Duration != 3.32 {
		t.Errorf("duration = %v, want 3.32", body.Duration)
	}
	if body.Text != cannedText {
		t.Errorf("text = %q, want %q", body.Text, cannedText)
	}
	if len(body.Segments) != 1 {
		t.Fatalf("segments = %d, want 1", len(body.Segments))
	}
	seg := body.Segments[0]
	if seg.ID != 0 || seg.Seek != 0 || seg.Start != 0.0 || seg.End != 3.32 {
		t.Errorf("segment timing = %+v, want id=0 seek=0 start=0 end=3.32", seg)
	}
	if len(seg.Tokens) != 3 || seg.Tokens[0] != 50364 || seg.Tokens[1] != 440 || seg.Tokens[2] != 7534 {
		t.Errorf("tokens = %v, want [50364 440 7534]", seg.Tokens)
	}
	if seg.Temperature != 0.0 {
		t.Errorf("temperature = %v, want 0", seg.Temperature)
	}
	if seg.AvgLogprob != -0.2860786020755768 {
		t.Errorf("avg_logprob = %v, want -0.2860786020755768", seg.AvgLogprob)
	}
	if seg.CompressionRatio != 0 {
		t.Errorf("compression_ratio = %v, want 0", seg.CompressionRatio)
	}
	if seg.NoSpeechProb != 0.00985979475080967 {
		t.Errorf("no_speech_prob = %v, want 0.00985979475080967", seg.NoSpeechProb)
	}
}

func TestTranslation(t *testing.T) {
	fake, state := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	req := multipartRequest(t, "/v1/audio/translations", map[string]string{
		"model":           "whisper-1",
		"response_format": "verbose_json",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var body struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body.Task != "translate" {
		t.Errorf("task = %q, want translate", body.Task)
	}
	fields, _, _ := state.snapshot()
	if fields["translate"] != "true" {
		t.Errorf("translate = %q, want true", fields["translate"])
	}
}

func TestTranscriptionText(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
		"model":           "whisper-1",
		"response_format": "text",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	// Upstream serves text as text/html; we must force text/plain.
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; charset=utf-8" {
		t.Errorf("content-type = %q, want text/plain; charset=utf-8", ct)
	}
	if rec.Body.String() != cannedText {
		t.Errorf("body = %q, want %q", rec.Body.String(), cannedText)
	}
}

func TestTranscriptionSubtitleFormats(t *testing.T) {
	cases := []struct {
		format string
		body   string
		ct     string
	}{
		{"srt", cannedSRT, "application/x-subrip"},
		{"vtt", cannedVTT, "text/vtt"},
	}
	for _, tc := range cases {
		t.Run(tc.format, func(t *testing.T) {
			fake, _ := newFakeWhisper(t)
			s := newTestServer(t, testConfig(), fake.URL)

			req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
				"model":           "whisper-1",
				"response_format": tc.format,
			}, "audio.wav", []byte("RIFF"))
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
			}
			if ct := rec.Header().Get("Content-Type"); ct != tc.ct {
				t.Errorf("content-type = %q, want %q", ct, tc.ct)
			}
			if rec.Body.String() != tc.body {
				t.Errorf("body = %q, want %q", rec.Body.String(), tc.body)
			}
		})
	}
}

func TestTranscriptionErrors(t *testing.T) {
	cases := []struct {
		name      string
		fields    map[string]string
		withFile  bool
		wantParam string
	}{
		{"missing file", map[string]string{"model": "whisper-1"}, false, "file"},
		{"wrong model", map[string]string{"model": "gpt-4o-transcribe"}, true, "model"},
		{"bad response format", map[string]string{"model": "whisper-1", "response_format": "tsv"}, true, "response_format"},
		{"bad temperature", map[string]string{"model": "whisper-1", "temperature": "hot"}, true, "temperature"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake, _ := newFakeWhisper(t)
			s := newTestServer(t, testConfig(), fake.URL)

			filename := ""
			if tc.withFile {
				filename = "audio.wav"
			}
			req := multipartRequest(t, "/v1/audio/transcriptions", tc.fields, filename, []byte("RIFF"))
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (body %s)", rec.Code, rec.Body.String())
			}
			var body struct {
				Error struct {
					Message string  `json:"message"`
					Type    string  `json:"type"`
					Param   *string `json:"param"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if body.Error.Type != "invalid_request_error" {
				t.Errorf("type = %q, want invalid_request_error", body.Error.Type)
			}
			if body.Error.Param == nil || *body.Error.Param != tc.wantParam {
				t.Errorf("param = %v, want %q", body.Error.Param, tc.wantParam)
			}
		})
	}
}

func TestUpstreamError(t *testing.T) {
	fake, state := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)
	state.setFail(true)

	req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
		"model": "whisper-1",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (body %s)", rec.Code, rec.Body.String())
	}
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body.Error.Type != "server_error" {
		t.Errorf("type = %q, want server_error", body.Error.Type)
	}
	if body.Error.Message == "" {
		t.Error("message should not be empty")
	}
}

func TestWriteError(t *testing.T) {
	t.Run("with param", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writeError(rec, http.StatusBadRequest, "invalid_request_error", "Invalid value for 'model'.", "model")
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", rec.Code)
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Errorf("content-type = %q, want application/json", ct)
		}
		var raw map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		errObj, ok := raw["error"].(map[string]any)
		if !ok {
			t.Fatalf("no error object: %s", rec.Body.String())
		}
		if errObj["message"] != "Invalid value for 'model'." {
			t.Errorf("message = %v", errObj["message"])
		}
		if errObj["type"] != "invalid_request_error" {
			t.Errorf("type = %v", errObj["type"])
		}
		if errObj["param"] != "model" {
			t.Errorf("param = %v, want model", errObj["param"])
		}
		if _, present := errObj["code"]; present {
			t.Errorf("code should be omitted when null: %s", rec.Body.String())
		}
	})

	t.Run("without param", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writeError(rec, http.StatusUnauthorized, "authentication_error", "Invalid API key", "")
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", rec.Code)
		}
		var raw map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		errObj, ok := raw["error"].(map[string]any)
		if !ok {
			t.Fatalf("no error object: %s", rec.Body.String())
		}
		if _, present := errObj["param"]; present {
			t.Errorf("param should be omitted when empty: %s", rec.Body.String())
		}
		if _, present := errObj["code"]; present {
			t.Errorf("code should be omitted when null: %s", rec.Body.String())
		}
	})
}
