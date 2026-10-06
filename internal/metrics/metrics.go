// Package metrics defines the Prometheus metrics exposed by whisper-oai.
//
// All collectors are registered on a private prometheus.Registry (plus the
// standard Go runtime and process collectors) and served in Prometheus text
// format from the /metrics endpoint. Every observation method is nil-safe:
// calling a method on a nil *Metrics is a no-op, so callers can invoke
// s.metrics.Observe(...) without guarding against metrics being disabled.
package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// namespace prefixes every metric name exposed by whisper-oai.
const namespace = "whisper_oai"

// durationBuckets is used for request/inference/ffmpeg duration histograms.
// Transcription can take several minutes, so the upper buckets are wide.
var durationBuckets = []float64{
	0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300,
}

// byteBuckets is used for request/response/upload size histograms (bytes).
var byteBuckets = []float64{
	1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
}

// Metrics holds every Prometheus collector for whisper-oai.
type Metrics struct {
	registry *prometheus.Registry

	// Overall API (HTTP) metrics.
	httpRequestsTotal    *prometheus.CounterVec
	httpRequestDuration  *prometheus.HistogramVec
	httpRequestsInFlight prometheus.Gauge
	httpRequestBytes     *prometheus.HistogramVec

	// whisper.cpp (whisper-server) inference metrics.
	inferenceRequestsTotal *prometheus.CounterVec
	inferenceDuration      *prometheus.HistogramVec
	upstreamInference      prometheus.Histogram
	transcribedSeconds     *prometheus.CounterVec
	uploadBytes            *prometheus.HistogramVec

	// whisper-server process health metrics.
	whisperServerUp       prometheus.Gauge
	whisperProcessRunning prometheus.Gauge
	whisperStartDuration  prometheus.Gauge

	// ffmpeg audio-extraction metrics.
	ffmpegExtractionsTotal *prometheus.CounterVec
	ffmpegDuration         prometheus.Histogram
	ffmpegInFlight         prometheus.Gauge
}

// New builds a Metrics with all collectors registered on a fresh registry
// (plus the standard Go runtime and process collectors).
func New() *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector())
	reg.MustRegister(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	m := &Metrics{
		registry: reg,

		httpRequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "http_requests_total",
			Help:      "Total number of HTTP requests received by the whisper-oai API, by method, route, and status code.",
		}, []string{"method", "path", "status"}),
		httpRequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "http_request_duration_seconds",
			Help:      "End-to-end HTTP request duration in seconds, by method and route.",
			Buckets:   durationBuckets,
		}, []string{"method", "path"}),
		httpRequestsInFlight: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "http_requests_in_flight",
			Help:      "Number of HTTP requests currently being served by the whisper-oai API.",
		}),
		httpRequestBytes: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "http_request_bytes",
			Help:      "HTTP request and response body sizes in bytes, by direction and route.",
			Buckets:   byteBuckets,
		}, []string{"direction", "path"}),

		inferenceRequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "inference_requests_total",
			Help:      "Total number of inference requests forwarded to whisper-server, by task and outcome.",
		}, []string{"task", "status"}),
		inferenceDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "inference_duration_seconds",
			Help:      "Duration of inference requests (upload spooling through whisper-server response) in seconds, by task.",
			Buckets:   durationBuckets,
		}, []string{"task"}),
		upstreamInference: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "upstream_inference_duration_seconds",
			Help:      "Time spent inside whisper-server per inference request, in seconds.",
			Buckets:   durationBuckets,
		}),
		transcribedSeconds: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "transcribed_seconds_total",
			Help:      "Total duration of audio content transcribed, in seconds, by task.",
		}, []string{"task"}),
		uploadBytes: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "upload_bytes",
			Help:      "Size of uploaded media files in bytes, by kind (audio or video).",
			Buckets:   byteBuckets,
		}, []string{"kind"}),

		whisperServerUp: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "whisper_server_up",
			Help:      "Whether the whisper-server backend answered /health (1 = up, 0 = down).",
		}),
		whisperProcessRunning: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "whisper_server_process_running",
			Help:      "Whether whisper-oai is managing a local whisper-server child process (1 = running, 0 = not). Always 0 in remote backend mode.",
		}),
		whisperStartDuration: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "whisper_server_start_duration_seconds",
			Help:      "Time taken for whisper-server to become ready after startup, in seconds.",
		}),

		ffmpegExtractionsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "ffmpeg_extractions_total",
			Help:      "Total number of ffmpeg audio extractions, by outcome.",
		}, []string{"status"}),
		ffmpegDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "ffmpeg_extraction_duration_seconds",
			Help:      "Duration of ffmpeg audio extractions from video uploads, in seconds.",
			Buckets:   durationBuckets,
		}),
		ffmpegInFlight: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "ffmpeg_extractions_in_flight",
			Help:      "Number of ffmpeg audio extractions currently running.",
		}),
	}

	reg.MustRegister(
		m.httpRequestsTotal,
		m.httpRequestDuration,
		m.httpRequestsInFlight,
		m.httpRequestBytes,
		m.inferenceRequestsTotal,
		m.inferenceDuration,
		m.upstreamInference,
		m.transcribedSeconds,
		m.uploadBytes,
		m.whisperServerUp,
		m.whisperProcessRunning,
		m.whisperStartDuration,
		m.ffmpegExtractionsTotal,
		m.ffmpegDuration,
		m.ffmpegInFlight,
	)
	return m
}

// Handler returns the http.Handler that serves the metrics in Prometheus
// text format (for the /metrics endpoint).
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

// Registry returns the underlying registry (for tests and advanced use).
func (m *Metrics) Registry() *prometheus.Registry {
	return m.registry
}

// ObserveHTTPRequest records one completed HTTP request. path should be the
// mux route pattern (e.g. "POST /v1/audio/transcriptions") to keep label
// cardinality low. reqBytes/respBytes are observed when >= 0.
func (m *Metrics) ObserveHTTPRequest(method, path string, status int, duration time.Duration, reqBytes, respBytes int64) {
	if m == nil {
		return
	}
	m.httpRequestsTotal.WithLabelValues(method, path, strconv.Itoa(status)).Inc()
	m.httpRequestDuration.WithLabelValues(method, path).Observe(duration.Seconds())
	if reqBytes >= 0 {
		m.httpRequestBytes.WithLabelValues("request", path).Observe(float64(reqBytes))
	}
	if respBytes >= 0 {
		m.httpRequestBytes.WithLabelValues("response", path).Observe(float64(respBytes))
	}
}

// IncInFlight increments the in-flight HTTP request gauge.
func (m *Metrics) IncInFlight() {
	if m == nil {
		return
	}
	m.httpRequestsInFlight.Inc()
}

// DecInFlight decrements the in-flight HTTP request gauge.
func (m *Metrics) DecInFlight() {
	if m == nil {
		return
	}
	m.httpRequestsInFlight.Dec()
}

// ObserveInference records one inference request forwarded to whisper-server.
// task is "transcribe" or "translate"; status is "success" or "error".
func (m *Metrics) ObserveInference(task, status string, duration time.Duration) {
	if m == nil {
		return
	}
	m.inferenceRequestsTotal.WithLabelValues(task, status).Inc()
	m.inferenceDuration.WithLabelValues(task).Observe(duration.Seconds())
}

// ObserveUpstreamInference records the time spent inside whisper-server for
// one inference request.
func (m *Metrics) ObserveUpstreamInference(duration time.Duration) {
	if m == nil {
		return
	}
	m.upstreamInference.Observe(duration.Seconds())
}

// ObserveTranscribedSeconds adds the duration of the audio that was
// transcribed (from the verbose_json response) to the throughput counter.
func (m *Metrics) ObserveTranscribedSeconds(task string, seconds float64) {
	if m == nil || seconds <= 0 {
		return
	}
	m.transcribedSeconds.WithLabelValues(task).Add(seconds)
}

// ObserveUploadBytes records the size of an uploaded media file. kind is
// "audio" or "video".
func (m *Metrics) ObserveUploadBytes(kind string, n int64) {
	if m == nil || n < 0 {
		return
	}
	m.uploadBytes.WithLabelValues(kind).Observe(float64(n))
}

// SetWhisperServerUp sets the whisper-server health gauge (1 = /health OK).
func (m *Metrics) SetWhisperServerUp(up bool) {
	if m == nil {
		return
	}
	if up {
		m.whisperServerUp.Set(1)
	} else {
		m.whisperServerUp.Set(0)
	}
}

// SetWhisperProcessRunning sets the local child-process liveness gauge.
func (m *Metrics) SetWhisperProcessRunning(running bool) {
	if m == nil {
		return
	}
	if running {
		m.whisperProcessRunning.Set(1)
	} else {
		m.whisperProcessRunning.Set(0)
	}
}

// SetWhisperStartDuration records how long whisper-server took to become
// ready after startup.
func (m *Metrics) SetWhisperStartDuration(d time.Duration) {
	if m == nil {
		return
	}
	m.whisperStartDuration.Set(d.Seconds())
}

// IncFFmpegInFlight increments the in-flight ffmpeg extraction gauge.
func (m *Metrics) IncFFmpegInFlight() {
	if m == nil {
		return
	}
	m.ffmpegInFlight.Inc()
}

// DecFFmpegInFlight decrements the in-flight ffmpeg extraction gauge.
func (m *Metrics) DecFFmpegInFlight() {
	if m == nil {
		return
	}
	m.ffmpegInFlight.Dec()
}

// ObserveFFmpegExtraction records one ffmpeg audio extraction. status is
// "success" or "error".
func (m *Metrics) ObserveFFmpegExtraction(status string, duration time.Duration) {
	if m == nil {
		return
	}
	m.ffmpegExtractionsTotal.WithLabelValues(status).Inc()
	m.ffmpegDuration.Observe(duration.Seconds())
}
