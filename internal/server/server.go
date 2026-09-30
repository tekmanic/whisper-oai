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
	"github.com/tekmanic/whisper-oai/internal/whisper"
)

// Server is the OpenAI-compatible HTTP server. It exposes the /v1/audio/*
// and /v1/models routes and forwards inference to the whisper-server managed
// by the given whisper.Manager.
type Server struct {
	cfg  *config.Config
	mgr  *whisper.Manager
	cli  *whisper.Client
	http *http.Server
	log  *slog.Logger
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

// Handler returns the http.Handler (mux) — exposed for tests.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/audio/transcriptions", s.handleTranscription)
	mux.HandleFunc("POST /v1/audio/translations", s.handleTranslation)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1/models/{model}", s.handleModel)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	return authMiddleware(s.cfg.Server.APIKey, mux)
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
