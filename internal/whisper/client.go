package whisper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
)

// Client is an HTTP client for the whisper-server API.
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient builds a client for the whisper-server at baseURL (e.g.
// "http://127.0.0.1:8080"). A trailing slash is tolerated.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{},
	}
}

// InferRequest describes a single /inference call.
type InferRequest struct {
	File           io.Reader // audio bytes
	Filename       string    // e.g. "audio.wav"
	Language       string    // "" = server default
	Prompt         string
	Temperature    float64
	Translate      bool
	ResponseFormat string // "json"|"text"|"srt"|"vtt"|"verbose_json"
}

// Segment is one transcription segment (whisper verbose_json shape).
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

// InferResponse is the parsed result of an /inference call.
type InferResponse struct {
	Text        string    // populated for json/verbose_json
	Language    string    // populated for verbose_json
	Duration    float64   // populated for verbose_json
	Segments    []Segment // populated for verbose_json
	RawBody     []byte    // populated for text/srt/vtt (passthrough)
	ContentType string    // content type of the upstream response
}

// Infer POSTs a multipart request to /inference. The multipart field names
// are EXACTLY: file, temperature, response_format, language, prompt,
// translate (translate is lowercase "true"/"false"). language and prompt are
// omitted when empty so the server falls back to its startup defaults. The
// response is parsed per ResponseFormat (json → {text}; verbose_json →
// task/language/duration/text/segments; text/srt/vtt → RawBody passthrough)
// and returned as a populated *InferResponse. Non-2xx responses return an
// error that includes the upstream body.
func (c *Client) Infer(ctx context.Context, req InferRequest) (*InferResponse, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	fw, err := w.CreateFormFile("file", req.Filename)
	if err != nil {
		return nil, fmt.Errorf("create file field: %w", err)
	}
	if _, err := io.Copy(fw, req.File); err != nil {
		return nil, fmt.Errorf("write file field: %w", err)
	}
	if err := w.WriteField("temperature", strconv.FormatFloat(req.Temperature, 'f', -1, 64)); err != nil {
		return nil, fmt.Errorf("write temperature field: %w", err)
	}
	if err := w.WriteField("response_format", req.ResponseFormat); err != nil {
		return nil, fmt.Errorf("write response_format field: %w", err)
	}
	if req.Language != "" {
		if err := w.WriteField("language", req.Language); err != nil {
			return nil, fmt.Errorf("write language field: %w", err)
		}
	}
	if req.Prompt != "" {
		if err := w.WriteField("prompt", req.Prompt); err != nil {
			return nil, fmt.Errorf("write prompt field: %w", err)
		}
	}
	if err := w.WriteField("translate", strconv.FormatBool(req.Translate)); err != nil {
		return nil, fmt.Errorf("write translate field: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("finish multipart body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/inference", &buf)
	if err != nil {
		return nil, fmt.Errorf("build inference request: %w", err)
	}
	httpReq.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("inference request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read inference response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("whisper-server returned %d: %s", resp.StatusCode, body)
	}

	out := &InferResponse{ContentType: resp.Header.Get("Content-Type")}
	switch req.ResponseFormat {
	case "verbose_json":
		var vj struct {
			Task     string    `json:"task"`
			Language string    `json:"language"`
			Duration float64   `json:"duration"`
			Text     string    `json:"text"`
			Segments []Segment `json:"segments"`
		}
		if err := json.Unmarshal(body, &vj); err != nil {
			return nil, fmt.Errorf("parse verbose_json response: %w", err)
		}
		out.Text = vj.Text
		out.Language = vj.Language
		out.Duration = vj.Duration
		out.Segments = vj.Segments
		for i := range out.Segments {
			// whisper-server omits these fields; normalize to zero.
			out.Segments[i].CompressionRatio = 0
			out.Segments[i].Seek = 0
		}
	case "text", "srt", "vtt":
		out.RawBody = body
	default: // "json" (and empty → upstream default json)
		var j struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(body, &j); err != nil {
			return nil, fmt.Errorf("parse json response: %w", err)
		}
		out.Text = j.Text
	}
	return out, nil
}

// Health returns nil if /health is 200.
func (c *Client) Health(ctx context.Context) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("build health request: %w", err)
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("whisper-server not ready: status %d", resp.StatusCode)
	}
	return nil
}
