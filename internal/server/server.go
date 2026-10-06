// Package server implements the OpenAI-compatible HTTP API that proxies
// transcription requests to a whisper.cpp whisper-server process.
package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/tekmanic/whisper-oai/internal/config"
	"github.com/tekmanic/whisper-oai/internal/metrics"
	"github.com/tekmanic/whisper-oai/internal/whisper"
)

// Server is the OpenAI-compatible HTTP server. It exposes the /v1/audio/*
// and /v1/models routes and forwards inference to the whisper-server managed
// by the given whisper.Manager.
type Server struct {
	cfg     *config.Config
	mgr     *whisper.Manager
	cli     *whisper.Client
	http    *http.Server
	log     *slog.Logger
	metrics *metrics.Metrics
}

// New wires the HTTP server. cli is built from mgr.URL().
func New(cfg *config.Config, mgr *whisper.Manager, log *slog.Logger) *Server {
	return &Server{
		cfg: cfg,
		mgr: mgr,
		cli: whisper.NewClient(mgr.URL()),
		log: log,
	}
}

// SetMetrics attaches the Prometheus metrics collector. When nil (the
// default), no metrics are collected and the /metrics route is not
// registered. Call before Handler is first used.
func (s *Server) SetMetrics(m *metrics.Metrics) {
	s.metrics = m
}

// Handler returns the http.Handler (mux) — exposed for tests.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/audio/transcriptions", s.handleTranscription)
	mux.HandleFunc("POST /v1/audio/translations", s.handleTranslation)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1/models/{model}", s.handleModel)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	if s.metrics != nil {
		mux.Handle("GET /metrics", s.metrics.Handler())
	}
	var h http.Handler = authMiddleware(s.cfg.Server.APIKey, s.log, mux)
	if s.metrics != nil {
		h = metricsMiddleware(s.metrics, h)
	}
	return loggingMiddleware(s.log, h)
}

// Start runs http.Server on cfg.ListenAddr(); blocks until ctx is cancelled
// or a fatal error. On ctx cancel it calls Shutdown.
func (s *Server) Start(ctx context.Context) error {
	s.http = &http.Server{
		Addr:    s.cfg.ListenAddr(),
		Handler: s.Handler(),
	}
	errCh := make(chan error, 1)
	go func() {
		s.log.Info("http server listening", "addr", s.cfg.ListenAddr())
		errCh <- s.http.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := s.http.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown http server: %w", err)
		}
		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("run http server: %w", err)
	}
}

// Shutdown gracefully stops the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.http == nil {
		return nil
	}
	return s.http.Shutdown(ctx)
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(p)
	r.bytes += n
	return n, err
}

func loggingMiddleware(log *slog.Logger, next http.Handler) http.Handler {
	if log == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rw, r)
		if rw.status == 0 {
			rw.status = http.StatusOK
		}
		dur := time.Since(start)

		attrs := []any{
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.status,
			"duration_ms", dur.Milliseconds(),
			"bytes", rw.bytes,
			"remote_addr", r.RemoteAddr,
		}

		switch {
		case rw.status >= 500:
			log.Error("http request completed", attrs...)
		case rw.status >= 400:
			log.Warn("http request completed", attrs...)
		default:
			log.Debug("http request completed", attrs...)
		}
	})
}

// metricsMiddleware records per-request Prometheus metrics (in-flight gauge,
// request counter, duration histogram, body-size histograms). It sits inside
// the logging middleware so both see the same request lifecycle.
func metricsMiddleware(m *metrics.Metrics, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		m.IncInFlight()
		defer m.DecInFlight()

		rw := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rw, r)
		if rw.status == 0 {
			rw.status = http.StatusOK
		}

		// Use the mux route pattern (e.g. "GET /v1/models/{model}") as the
		// path label to keep cardinality bounded.
		path := r.Pattern
		if path == "" {
			path = r.URL.Path
		}
		m.ObserveHTTPRequest(r.Method, path, rw.status, time.Since(start), r.ContentLength, int64(rw.bytes))
	})
}
