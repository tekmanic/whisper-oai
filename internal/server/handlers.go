package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/tekmanic/whisper-oai/internal/audio"
	"github.com/tekmanic/whisper-oai/internal/whisper"
)

// validFormats is the set of response formats accepted on /v1/audio/*.
var validFormats = map[string]bool{
	"json":         true,
	"text":         true,
	"srt":          true,
	"verbose_json": true,
	"vtt":          true,
}

// modelCreated is the fixed "created" timestamp reported for the model,
// matching the OpenAI whisper-1 example.
const modelCreated = 1687882411

// modelInfo is a single OpenAI model object.
type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// modelList is the OpenAI model list envelope.
type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

// openAISegment is the OpenAI TranscriptionSegment shape.
type openAISegment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int64 `json:"tokens,omitempty"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}

// openAIVerboseJSON is the OpenAI verbose_json transcription response.
type openAIVerboseJSON struct {
	Task     string          `json:"task"`
	Language string          `json:"language"`
	Duration float64         `json:"duration"`
	Text     string          `json:"text"`
	Segments []openAISegment `json:"segments"`
}

// handleTranscription serves POST /v1/audio/transcriptions (translate=false).
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	s.handleInfer(w, r, false)
}

// handleTranslation serves POST /v1/audio/translations (translate=true).
func (s *Server) handleTranslation(w http.ResponseWriter, r *http.Request) {
	s.handleInfer(w, r, true)
}

// handleInfer parses the OpenAI multipart request, forwards it to
// whisper-server, and writes the response in the requested format.
func (s *Server) handleInfer(w http.ResponseWriter, r *http.Request, translate bool) {
	start := time.Now()
	s.log.Debug("received inference request",
		"method", r.Method,
		"path", r.URL.Path,
		"translate", translate,
		"remote_addr", r.RemoteAddr,
	)

	// Enforce the configured upload size limit (0 = unlimited).
	if s.cfg.Audio.MaxUploadMB > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, int64(s.cfg.Audio.MaxUploadMB)<<20)
	}

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			s.log.Warn("multipart form exceeded max upload size", "max_upload_mb", s.cfg.Audio.MaxUploadMB)
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error",
				fmt.Sprintf("upload exceeds the maximum size of %d MB", s.cfg.Audio.MaxUploadMB), "")
			return
		}
		s.log.Warn("failed to parse multipart form", "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error", "failed to parse multipart form", "")
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		s.log.Warn("request missing file field", "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error", "no 'file' field in the request", "file")
		return
	}
	defer file.Close()
	s.log.Debug("received upload", "filename", header.Filename, "content_type", header.Header.Get("Content-Type"))

	// Spool the upload to a temp file so large uploads stream to disk (not
	// memory) and video files can be addressed by path for ffmpeg.
	srcPath, err := audio.Spool(file, header.Filename, s.cfg.Audio.TempDir)
	if err != nil {
		s.log.Error("failed to spool upload", "error", err)
		writeError(w, http.StatusInternalServerError, "server_error", "failed to store the uploaded file", "")
		return
	}
	defer os.Remove(srcPath)

	// Video containers (mp4, mov, mkv, ...) cannot be decoded by
	// whisper-server; strip the audio track with ffmpeg first.
	var (
		inferFile io.Reader
		inferName = header.Filename
	)
	if audio.IsVideo(header.Filename, header.Header.Get("Content-Type")) {
		s.log.Info("video upload detected; extracting audio track", "file", header.Filename)
		wavPath, err := audio.Extract(r.Context(), srcPath, s.cfg.Audio.FfmpegBin, s.cfg.Audio.TempDir, s.log)
		if err != nil {
			s.log.Error("audio extraction failed", "error", err)
			writeError(w, http.StatusBadGateway, "server_error",
				fmt.Sprintf("audio extraction from video failed: %v", err), "")
			return
		}
		defer os.Remove(wavPath)
		wf, err := os.Open(wavPath)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "server_error", "failed to open extracted audio", "")
			return
		}
		defer wf.Close()
		inferFile, inferName = wf, filepath.Base(wavPath)
	} else {
		sf, err := os.Open(srcPath)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "server_error", "failed to open uploaded file", "")
			return
		}
		defer sf.Close()
		inferFile = sf
	}

	model := r.FormValue("model")
	if model != s.cfg.Server.ModelName {
		s.log.Warn("request used invalid model", "got", model, "expected", s.cfg.Server.ModelName)
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("Invalid value for 'model'. Expected %q.", s.cfg.Server.ModelName), "model")
		return
	}

	format := r.FormValue("response_format")
	if format == "" {
		format = "json"
	}
	if !validFormats[format] {
		s.log.Warn("request used invalid response format", "response_format", format)
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"Invalid value for 'response_format'. Expected one of: json, text, srt, verbose_json, vtt.",
			"response_format")
		return
	}

	temperature := 0.0
	if v := r.FormValue("temperature"); v != "" {
		temperature, err = strconv.ParseFloat(v, 64)
		if err != nil {
			s.log.Warn("request used invalid temperature", "temperature", v)
			writeError(w, http.StatusBadRequest, "invalid_request_error", "Invalid value for 'temperature'.", "temperature")
			return
		}
	}

	s.log.Info("forwarding inference request",
		"filename", inferName,
		"translate", translate,
		"response_format", format,
		"language", r.FormValue("language"),
		"temperature", temperature,
	)

	req := whisper.InferRequest{
		File:           inferFile,
		Filename:       inferName,
		Language:       r.FormValue("language"),
		Prompt:         r.FormValue("prompt"),
		Temperature:    temperature,
		Translate:      translate,
		ResponseFormat: format,
	}

	resp, err := s.cli.Infer(r.Context(), req)
	if err != nil {
		s.log.Error("inference failed", "error", err)
		writeError(w, http.StatusBadGateway, "server_error", fmt.Sprintf("upstream inference failed: %v", err), "")
		return
	}
	s.log.Info("inference completed",
		"filename", inferName,
		"response_format", format,
		"text_len", len(resp.Text),
		"raw_len", len(resp.RawBody),
		"duration_ms", time.Since(start).Milliseconds(),
	)
	s.writeInferResponse(w, format, translate, resp)
}

// writeInferResponse maps a whisper.InferResponse to the OpenAI shape for the
// requested response format.
func (s *Server) writeInferResponse(w http.ResponseWriter, format string, translate bool, resp *whisper.InferResponse) {
	switch format {
	case "json":
		writeJSON(w, http.StatusOK, map[string]string{"text": resp.Text})
	case "verbose_json":
		task := "transcribe"
		if translate {
			task = "translate"
		}
		segments := make([]openAISegment, 0, len(resp.Segments))
		for _, seg := range resp.Segments {
			segments = append(segments, openAISegment{
				ID:               seg.ID,
				Seek:             seg.Seek,
				Start:            seg.Start,
				End:              seg.End,
				Text:             seg.Text,
				Tokens:           seg.Tokens,
				Temperature:      seg.Temperature,
				AvgLogprob:       seg.AvgLogprob,
				CompressionRatio: seg.CompressionRatio,
				NoSpeechProb:     seg.NoSpeechProb,
			})
		}
		writeJSON(w, http.StatusOK, openAIVerboseJSON{
			Task:     task,
			Language: resp.Language,
			Duration: resp.Duration,
			Text:     resp.Text,
			Segments: segments,
		})
	case "text", "srt", "vtt":
		contentType := resp.ContentType
		if format == "text" {
			// whisper-server returns text/html for text (upstream quirk);
			// OpenAI clients expect text/plain.
			contentType = "text/plain; charset=utf-8"
		}
		if contentType == "" {
			contentType = "text/plain; charset=utf-8"
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(resp.RawBody)
	}
}

// handleModels serves GET /v1/models.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.log.Debug("serving model list")
	writeJSON(w, http.StatusOK, modelList{
		Object: "list",
		Data:   []modelInfo{s.modelInfo()},
	})
}

// handleModel serves GET /v1/models/{model}; 404 if the id is not the
// configured model.
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("model")
	if id != s.cfg.Server.ModelName {
		s.log.Warn("requested unknown model", "model", id)
		writeError(w, http.StatusNotFound, "invalid_request_error",
			fmt.Sprintf("The model '%s' does not exist", id), "model")
		return
	}
	s.log.Debug("serving model details", "model", id)
	writeJSON(w, http.StatusOK, s.modelInfo())
}

// modelInfo returns the synthesized model object for the configured model.
func (s *Server) modelInfo() modelInfo {
	return modelInfo{
		ID:      s.cfg.Server.ModelName,
		Object:  "model",
		Created: modelCreated,
		OwnedBy: "whisper-oai",
	}
}

// handleHealthz serves GET /healthz (proxy liveness).
func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	s.log.Debug("serving health check")
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// writeJSON writes v as an application/json response with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// A failed encode means the client went away; nothing useful to do.
	_ = json.NewEncoder(w).Encode(v)
}
