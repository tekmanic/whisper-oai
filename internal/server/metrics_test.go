package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tekmanic/whisper-oai/internal/config"
	"github.com/tekmanic/whisper-oai/internal/metrics"
)

// scrapeMetrics serves GET /metrics on the given server and returns the body.
func scrapeMetrics(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics: status = %d, want 200", rec.Code)
	}
	return rec.Body.String()
}

func TestMetricsEndpoint(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)
	s.SetMetrics(metrics.New())

	// Generate some traffic first.
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	body := scrapeMetrics(t, s)
	for _, want := range []string{
		"whisper_oai_http_requests_total",
		`path="GET /healthz"`,
		"whisper_oai_http_request_duration_seconds",
		"whisper_oai_http_requests_in_flight",
		"whisper_oai_whisper_server_up",
		"whisper_oai_ffmpeg_extractions_in_flight",
		"go_goroutines",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}

func TestMetricsEndpointRequiresAuth(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	cfg := testConfig()
	cfg.Server.APIKey = "secret"
	s := newTestServer(t, cfg, fake.URL)
	s.SetMetrics(metrics.New())

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("no key: status = %d, want 401", rec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("with key: status = %d, want 200", rec.Code)
	}
}

func TestMetricsDisabledByDefault(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 when metrics are not enabled", rec.Code)
	}
}

func TestInferenceMetrics(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)
	s.SetMetrics(metrics.New())

	req := multipartRequest(t, "/v1/audio/transcriptions", map[string]string{
		"model":           "whisper-1",
		"response_format": "verbose_json",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}

	body := scrapeMetrics(t, s)
	for _, want := range []string{
		`whisper_oai_inference_requests_total{status="success",task="transcribe"} 1`,
		`whisper_oai_inference_duration_seconds_count{task="transcribe"} 1`,
		`whisper_oai_upstream_inference_duration_seconds_count 1`,
		`whisper_oai_transcribed_seconds_total{task="transcribe"} 3.32`,
		`whisper_oai_upload_bytes_count{kind="audio"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}

func TestInferenceErrorMetrics(t *testing.T) {
	fake, state := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL)
	s.SetMetrics(metrics.New())
	state.setFail(true)

	req := multipartRequest(t, "/v1/audio/translations", map[string]string{
		"model": "whisper-1",
	}, "audio.wav", []byte("RIFF"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}

	body := scrapeMetrics(t, s)
	for _, want := range []string{
		`whisper_oai_inference_requests_total{status="error",task="translate"} 1`,
		`whisper_oai_http_requests_total{method="POST",path="POST /v1/audio/translations",status="502"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}

func TestVideoUploadFFmpegMetrics(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	script := writeFakeFFmpeg(t)

	cfg := testConfig()
	cfg.Audio = config.AudioConfig{FfmpegBin: script}
	s := newTestServer(t, cfg, fake.URL)
	s.SetMetrics(metrics.New())

	req := multipartRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "whisper-1", "response_format": "json"},
		"movie.mp4", []byte("fake mp4 bytes"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}

	body := scrapeMetrics(t, s)
	for _, want := range []string{
		`whisper_oai_ffmpeg_extractions_total{status="success"} 1`,
		`whisper_oai_ffmpeg_extraction_duration_seconds_count 1`,
		`whisper_oai_upload_bytes_count{kind="video"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}

func TestVideoUploadFFmpegErrorMetrics(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	script := writeFailingFFmpeg(t)

	cfg := testConfig()
	cfg.Audio = config.AudioConfig{FfmpegBin: script}
	s := newTestServer(t, cfg, fake.URL)
	s.SetMetrics(metrics.New())

	req := multipartRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "whisper-1"},
		"movie.mp4", []byte("fake mp4 bytes"))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}

	body := scrapeMetrics(t, s)
	for _, want := range []string{
		`whisper_oai_ffmpeg_extractions_total{status="error"} 1`,
		`whisper_oai_http_requests_total{method="POST",path="POST /v1/audio/transcriptions",status="502"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}
