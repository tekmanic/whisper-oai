package metrics

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

// histogramCount returns the sample count of the histogram series identified
// by name and labels (0 if the series is absent).
func histogramCount(t *testing.T, m *Metrics, name string, labels map[string]string) uint64 {
	t.Helper()
	mfs, err := m.registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, mtr := range mf.GetMetric() {
			if labelMatch(mtr.GetLabel(), labels) {
				return mtr.GetHistogram().GetSampleCount()
			}
		}
	}
	return 0
}

func labelMatch(got []*dto.LabelPair, want map[string]string) bool {
	if len(got) != len(want) {
		return false
	}
	for _, lp := range got {
		if want[lp.GetName()] != lp.GetValue() {
			return false
		}
	}
	return true
}

func TestNewRegistersAllMetrics(t *testing.T) {
	m := New()
	// Vec collectors emit no series until a label combination is observed,
	// so touch each one before gathering.
	m.ObserveHTTPRequest("GET", "GET /healthz", 200, time.Millisecond, 0, 0)
	m.ObserveInference("transcribe", "success", time.Millisecond)
	m.ObserveTranscribedSeconds("transcribe", 1.0)
	m.ObserveUploadBytes("audio", 10)
	m.ObserveFFmpegExtraction("success", time.Millisecond)

	mfs, err := m.registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	names := map[string]bool{}
	for _, mf := range mfs {
		names[mf.GetName()] = true
	}
	want := []string{
		// Overall API
		"whisper_oai_http_requests_total",
		"whisper_oai_http_request_duration_seconds",
		"whisper_oai_http_requests_in_flight",
		"whisper_oai_http_request_bytes",
		// whisper.cpp inference
		"whisper_oai_inference_requests_total",
		"whisper_oai_inference_duration_seconds",
		"whisper_oai_upstream_inference_duration_seconds",
		"whisper_oai_transcribed_seconds_total",
		"whisper_oai_upload_bytes",
		// whisper-server process
		"whisper_oai_whisper_server_up",
		"whisper_oai_whisper_server_process_running",
		"whisper_oai_whisper_server_start_duration_seconds",
		// ffmpeg
		"whisper_oai_ffmpeg_extractions_total",
		"whisper_oai_ffmpeg_extraction_duration_seconds",
		"whisper_oai_ffmpeg_extractions_in_flight",
		// Standard collectors
		"go_goroutines",
		"process_cpu_seconds_total",
	}
	for _, w := range want {
		if !names[w] {
			t.Errorf("metric %q not registered", w)
		}
	}
}

func TestObserveHTTPRequest(t *testing.T) {
	m := New()
	const path = "POST /v1/audio/transcriptions"
	m.ObserveHTTPRequest("POST", path, 200, 1500*time.Millisecond, 1024, 42)
	m.ObserveHTTPRequest("POST", path, 502, 50*time.Millisecond, 1024, 10)
	// Negative byte counts (unknown length) must not be observed.
	m.ObserveHTTPRequest("GET", "GET /healthz", 200, time.Millisecond, -1, -1)

	if got := testutil.ToFloat64(m.httpRequestsTotal.WithLabelValues("POST", path, "200")); got != 1 {
		t.Errorf("requests_total{200} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.httpRequestsTotal.WithLabelValues("POST", path, "502")); got != 1 {
		t.Errorf("requests_total{502} = %v, want 1", got)
	}
	if got := histogramCount(t, m, "whisper_oai_http_request_duration_seconds", map[string]string{"method": "POST", "path": path}); got != 2 {
		t.Errorf("request_duration count = %v, want 2", got)
	}
	if got := histogramCount(t, m, "whisper_oai_http_request_bytes", map[string]string{"direction": "request", "path": path}); got != 2 {
		t.Errorf("request_bytes{request} count = %v, want 2", got)
	}
	if got := histogramCount(t, m, "whisper_oai_http_request_bytes", map[string]string{"direction": "response", "path": path}); got != 2 {
		t.Errorf("request_bytes{response} count = %v, want 2", got)
	}
	if got := histogramCount(t, m, "whisper_oai_http_request_bytes", map[string]string{"direction": "request", "path": "GET /healthz"}); got != 0 {
		t.Errorf("request_bytes{healthz} count = %v, want 0 (negative skipped)", got)
	}
}

func TestInFlightGauge(t *testing.T) {
	m := New()
	m.IncInFlight()
	m.IncInFlight()
	if got := testutil.ToFloat64(m.httpRequestsInFlight); got != 2 {
		t.Errorf("in_flight = %v, want 2", got)
	}
	m.DecInFlight()
	if got := testutil.ToFloat64(m.httpRequestsInFlight); got != 1 {
		t.Errorf("in_flight = %v, want 1", got)
	}
}

func TestInferenceAndFFmpegObservation(t *testing.T) {
	m := New()
	m.ObserveInference("transcribe", "success", time.Second)
	m.ObserveInference("transcribe", "error", 2*time.Second)
	m.ObserveInference("translate", "success", 3*time.Second)
	m.ObserveUpstreamInference(800 * time.Millisecond)
	m.ObserveTranscribedSeconds("transcribe", 3.32)
	m.ObserveTranscribedSeconds("transcribe", 0) // must be ignored
	m.ObserveUploadBytes("video", 1<<20)
	m.ObserveUploadBytes("audio", 1024)

	if got := testutil.ToFloat64(m.inferenceRequestsTotal.WithLabelValues("transcribe", "success")); got != 1 {
		t.Errorf("inference{transcribe,success} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.inferenceRequestsTotal.WithLabelValues("transcribe", "error")); got != 1 {
		t.Errorf("inference{transcribe,error} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.inferenceRequestsTotal.WithLabelValues("translate", "success")); got != 1 {
		t.Errorf("inference{translate,success} = %v, want 1", got)
	}
	if got := histogramCount(t, m, "whisper_oai_inference_duration_seconds", map[string]string{"task": "transcribe"}); got != 2 {
		t.Errorf("inference_duration{transcribe} count = %v, want 2", got)
	}
	if got := histogramCount(t, m, "whisper_oai_upstream_inference_duration_seconds", nil); got != 1 {
		t.Errorf("upstream_inference count = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.transcribedSeconds.WithLabelValues("transcribe")); got != 3.32 {
		t.Errorf("transcribed_seconds = %v, want 3.32", got)
	}
	if got := histogramCount(t, m, "whisper_oai_upload_bytes", map[string]string{"kind": "video"}); got != 1 {
		t.Errorf("upload_bytes{video} count = %v, want 1", got)
	}
	if got := histogramCount(t, m, "whisper_oai_upload_bytes", map[string]string{"kind": "audio"}); got != 1 {
		t.Errorf("upload_bytes{audio} count = %v, want 1", got)
	}

	m.IncFFmpegInFlight()
	if got := testutil.ToFloat64(m.ffmpegInFlight); got != 1 {
		t.Errorf("ffmpeg_in_flight = %v, want 1", got)
	}
	m.DecFFmpegInFlight()
	m.ObserveFFmpegExtraction("success", 500*time.Millisecond)
	m.ObserveFFmpegExtraction("error", 100*time.Millisecond)
	if got := testutil.ToFloat64(m.ffmpegExtractionsTotal.WithLabelValues("success")); got != 1 {
		t.Errorf("ffmpeg{success} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.ffmpegExtractionsTotal.WithLabelValues("error")); got != 1 {
		t.Errorf("ffmpeg{error} = %v, want 1", got)
	}
	if got := histogramCount(t, m, "whisper_oai_ffmpeg_extraction_duration_seconds", nil); got != 2 {
		t.Errorf("ffmpeg_duration count = %v, want 2", got)
	}
}

func TestProcessGauges(t *testing.T) {
	m := New()
	m.SetWhisperServerUp(true)
	if got := testutil.ToFloat64(m.whisperServerUp); got != 1 {
		t.Errorf("whisper_server_up = %v, want 1", got)
	}
	m.SetWhisperServerUp(false)
	if got := testutil.ToFloat64(m.whisperServerUp); got != 0 {
		t.Errorf("whisper_server_up = %v, want 0", got)
	}
	m.SetWhisperProcessRunning(true)
	if got := testutil.ToFloat64(m.whisperProcessRunning); got != 1 {
		t.Errorf("process_running = %v, want 1", got)
	}
	m.SetWhisperStartDuration(42 * time.Second)
	if got := testutil.ToFloat64(m.whisperStartDuration); got != 42 {
		t.Errorf("start_duration = %v, want 42", got)
	}
}

func TestNilSafe(t *testing.T) {
	var m *Metrics
	m.ObserveHTTPRequest("GET", "GET /healthz", 200, time.Millisecond, 0, 0)
	m.IncInFlight()
	m.DecInFlight()
	m.ObserveInference("transcribe", "success", time.Millisecond)
	m.ObserveUpstreamInference(time.Millisecond)
	m.ObserveTranscribedSeconds("transcribe", 1.0)
	m.ObserveUploadBytes("audio", 10)
	m.SetWhisperServerUp(true)
	m.SetWhisperProcessRunning(true)
	m.SetWhisperStartDuration(time.Second)
	m.IncFFmpegInFlight()
	m.DecFFmpegInFlight()
	m.ObserveFFmpegExtraction("success", time.Millisecond)
}

func TestHandlerServesPrometheus(t *testing.T) {
	m := New()
	m.SetWhisperServerUp(true)
	m.ObserveHTTPRequest("GET", "GET /healthz", 200, 5*time.Millisecond, 0, 15)

	ts := httptest.NewServer(m.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	for _, want := range []string{
		"whisper_oai_whisper_server_up 1",
		`whisper_oai_http_requests_total{method="GET",path="GET /healthz",status="200"} 1`,
		"go_goroutines",
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("metrics output missing %q", want)
		}
	}
}
