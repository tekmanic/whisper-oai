package whisper

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// capturedRequest holds the fields of the last /inference request seen by the
// fake server.
type capturedRequest struct {
	fields   map[string]string
	fileName string
	fileBody string
}

// newFakeServer starts a fake whisper-server. GET /health always returns 200;
// POST /inference replies with (status, contentType, body) and, if capture is
// non-nil, records the request's multipart fields and uploaded file.
func newFakeServer(t *testing.T, status int, contentType, body string, capture *capturedRequest) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/inference":
			if capture != nil {
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Errorf("parse multipart form: %v", err)
					return
				}
				capture.fields = map[string]string{}
				for k, v := range r.MultipartForm.Value {
					capture.fields[k] = v[0]
				}
				f, hdr, err := r.FormFile("file")
				if err != nil {
					t.Errorf("read file field: %v", err)
					return
				}
				capture.fileName = hdr.Filename
				b, rerr := io.ReadAll(f)
				f.Close()
				if rerr != nil {
					t.Errorf("read uploaded file: %v", rerr)
					return
				}
				capture.fileBody = string(b)
			}
			w.Header().Set("Content-Type", contentType)
			w.WriteHeader(status)
			io.WriteString(w, body)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(ts.Close)
	return ts
}

func TestInferJSON(t *testing.T) {
	var cap capturedRequest
	ts := newFakeServer(t, http.StatusOK, "application/json", `{"text":"hello world"}`, &cap)
	c := NewClient(ts.URL)

	resp, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("RIFF-fake-wav-bytes"),
		Filename:       "audio.wav",
		Language:       "en",
		Prompt:         "a prompt",
		Temperature:    0.5,
		Translate:      true,
		ResponseFormat: "json",
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.Text != "hello world" {
		t.Errorf("Text = %q, want %q", resp.Text, "hello world")
	}
	if resp.ContentType != "application/json" {
		t.Errorf("ContentType = %q, want application/json", resp.ContentType)
	}

	// Exact multipart field names and values.
	wantFields := map[string]string{
		"temperature":     "0.5",
		"response_format": "json",
		"language":        "en",
		"prompt":          "a prompt",
		"translate":       "true",
	}
	for k, v := range wantFields {
		if got := cap.fields[k]; got != v {
			t.Errorf("field %q = %q, want %q", k, got, v)
		}
	}
	if cap.fileName != "audio.wav" {
		t.Errorf("file name = %q, want audio.wav", cap.fileName)
	}
	if cap.fileBody != "RIFF-fake-wav-bytes" {
		t.Errorf("file body = %q", cap.fileBody)
	}
}

func TestInferOmitsEmptyOptionalFields(t *testing.T) {
	var cap capturedRequest
	ts := newFakeServer(t, http.StatusOK, "application/json", `{"text":""}`, &cap)
	c := NewClient(ts.URL)
	if _, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("x"),
		Filename:       "a.wav",
		Temperature:    0,
		ResponseFormat: "json",
	}); err != nil {
		t.Fatalf("Infer: %v", err)
	}
	for _, k := range []string{"language", "prompt"} {
		if _, ok := cap.fields[k]; ok {
			t.Errorf("field %q should be omitted when empty", k)
		}
	}
	if got := cap.fields["translate"]; got != "false" {
		t.Errorf("translate = %q, want lowercase \"false\"", got)
	}
	if got := cap.fields["temperature"]; got != "0" {
		t.Errorf("temperature = %q, want \"0\"", got)
	}
}

func TestInferVerboseJSON(t *testing.T) {
	body := `{"task":"transcribe","language":"english","duration":3.32,"text":"And so my fellow Americans.",
	 "segments":[
	   {"id":0,"text":" And so my fellow","start":0.0,"end":1.5,"tokens":[1,2,3],"temperature":0.0,"avg_logprob":-0.28,"no_speech_prob":0.01},
	   {"id":1,"text":" Americans.","start":1.5,"end":3.32,"tokens":[4,5],"temperature":0.0,"avg_logprob":-0.1,"no_speech_prob":0.02}
	 ]}`
	ts := newFakeServer(t, http.StatusOK, "application/json", body, nil)
	c := NewClient(ts.URL)

	resp, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("x"),
		Filename:       "a.wav",
		ResponseFormat: "verbose_json",
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.Language != "english" {
		t.Errorf("Language = %q, want english", resp.Language)
	}
	if resp.Duration != 3.32 {
		t.Errorf("Duration = %v, want 3.32", resp.Duration)
	}
	if resp.Text != "And so my fellow Americans." {
		t.Errorf("Text = %q", resp.Text)
	}
	if len(resp.Segments) != 2 {
		t.Fatalf("len(Segments) = %d, want 2", len(resp.Segments))
	}
	s0 := resp.Segments[0]
	if s0.ID != 0 || s0.Text != " And so my fellow" || s0.Start != 0.0 || s0.End != 1.5 {
		t.Errorf("Segments[0] = %+v", s0)
	}
	if len(s0.Tokens) != 3 || s0.Tokens[0] != 1 || s0.Tokens[2] != 3 {
		t.Errorf("Segments[0].Tokens = %v, want [1 2 3]", s0.Tokens)
	}
	if s0.Temperature != 0.0 || s0.AvgLogprob != -0.28 || s0.NoSpeechProb != 0.01 {
		t.Errorf("Segments[0] stats = %+v", s0)
	}
	if s0.CompressionRatio != 0 || s0.Seek != 0 {
		t.Errorf("Segments[0] CompressionRatio/Seek = %v/%d, want 0/0", s0.CompressionRatio, s0.Seek)
	}
	s1 := resp.Segments[1]
	if s1.ID != 1 || s1.Start != 1.5 || s1.End != 3.32 || len(s1.Tokens) != 2 {
		t.Errorf("Segments[1] = %+v", s1)
	}
}

func TestInferPassthrough(t *testing.T) {
	tests := []struct {
		format string
		body   string
		ct     string
	}{
		{"text", "hello\nworld", "text/html; charset=utf-8"},
		{"srt", "1\n00:00:00,000 --> 00:00:01,590\nhello", "application/x-subrip"},
		{"vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.590\nhello", "text/vtt"},
	}
	for _, tt := range tests {
		t.Run(tt.format, func(t *testing.T) {
			ts := newFakeServer(t, http.StatusOK, tt.ct, tt.body, nil)
			c := NewClient(ts.URL)
			resp, err := c.Infer(context.Background(), InferRequest{
				File:           strings.NewReader("x"),
				Filename:       "a.wav",
				ResponseFormat: tt.format,
			})
			if err != nil {
				t.Fatalf("Infer: %v", err)
			}
			if string(resp.RawBody) != tt.body {
				t.Errorf("RawBody = %q, want %q", resp.RawBody, tt.body)
			}
			if resp.ContentType != tt.ct {
				t.Errorf("ContentType = %q, want %q", resp.ContentType, tt.ct)
			}
			if resp.Text != "" {
				t.Errorf("Text = %q, want empty for passthrough", resp.Text)
			}
			if len(resp.Segments) != 0 {
				t.Errorf("Segments = %v, want empty for passthrough", resp.Segments)
			}
		})
	}
}

func TestInferUpstreamError(t *testing.T) {
	ts := newFakeServer(t, http.StatusInternalServerError, "application/json", `{"error":"failed to process audio"}`, nil)
	c := NewClient(ts.URL)
	_, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("x"),
		Filename:       "a.wav",
		ResponseFormat: "json",
	})
	if err == nil {
		t.Fatal("Infer: expected error for 500 response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should include the status code: %v", err)
	}
	if !strings.Contains(err.Error(), "failed to process audio") {
		t.Errorf("error should include the upstream body: %v", err)
	}
}

func TestInferUnparseableJSON(t *testing.T) {
	ts := newFakeServer(t, http.StatusOK, "application/json", `not-json`, nil)
	c := NewClient(ts.URL)
	_, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("x"),
		Filename:       "a.wav",
		ResponseFormat: "json",
	})
	if err == nil {
		t.Fatal("Infer: expected error for unparseable json body")
	}
}

func TestHealth(t *testing.T) {
	t.Run("ok", func(t *testing.T) {
		ts := newFakeServer(t, http.StatusOK, "", "", nil)
		c := NewClient(ts.URL)
		if err := c.Health(context.Background()); err != nil {
			t.Fatalf("Health: %v", err)
		}
	})
	t.Run("not ready", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"status":"loading model"}`)
		}))
		t.Cleanup(ts.Close)
		c := NewClient(ts.URL)
		if err := c.Health(context.Background()); err == nil {
			t.Fatal("Health: expected error for 503")
		}
	})
}

func TestNewClientTrimsTrailingSlash(t *testing.T) {
	ts := newFakeServer(t, http.StatusOK, "application/json", `{"text":"ok"}`, nil)
	c := NewClient(ts.URL + "/")
	resp, err := c.Infer(context.Background(), InferRequest{
		File:           strings.NewReader("x"),
		Filename:       "a.wav",
		ResponseFormat: "json",
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
}
