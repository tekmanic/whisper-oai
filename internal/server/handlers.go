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
	"strings"
	"time"

	"github.com/tekmanic/whisper-oai/internal/audio"
	"github.com/tekmanic/whisper-oai/internal/download"
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

// inferParams holds the per-request inference parameters, parsed from either
// the multipart form or the JSON body.
type inferParams struct {
	model       string
	format      string
	language    string
	prompt      string
	temperature float64
}

// jsonInferRequest is the JSON body accepted on /v1/audio/transcriptions and
// /v1/audio/translations for URL ingestion. All fields except url are
// optional and mirror the multipart form fields.
type jsonInferRequest struct {
	URL            string  `json:"url"`
	Model          string  `json:"model"`
	ResponseFormat string  `json:"response_format"`
	Language       string  `json:"language"`
	Prompt         string  `json:"prompt"`
	Temperature    float64 `json:"temperature"`
}

// handleInfer resolves the request's media (multipart upload or JSON url),
// prepares it, forwards it to whisper-server, and writes the response in the
// requested format.
func (s *Server) handleInfer(w http.ResponseWriter, r *http.Request, translate bool) {
	start := time.Now()
	s.log.Debug("received inference request",
		"method", r.Method,
		"path", r.URL.Path,
		"translate", translate,
		"remote_addr", r.RemoteAddr,
	)

	// Resolve the media source to a local file: a multipart file upload
	// (spooled to disk) or a JSON body with a "url" field (downloaded via
	// yt-dlp). On failure the error response is already written.
	srcPath, filename, contentType, params, ok := s.resolveSource(w, r)
	if !ok {
		return
	}
	defer os.Remove(srcPath)

	if params.model != s.cfg.Server.ModelName {
		s.log.Warn("request used invalid model", "got", params.model, "expected", s.cfg.Server.ModelName)
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("Invalid value for 'model'. Expected %q.", s.cfg.Server.ModelName), "model")
		return
	}

	if params.format == "" {
		params.format = "json"
	}
	if !validFormats[params.format] {
		s.log.Warn("request used invalid response format", "response_format", params.format)
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"Invalid value for 'response_format'. Expected one of: json, text, srt, verbose_json, vtt.",
			"response_format")
		return
	}

	// Video containers (mp4, mov, mkv, ...) cannot be decoded by
	// whisper-server; strip the audio track with ffmpeg first.
	var (
		inferFile io.Reader
		inferName = filename
	)
	isVideo := audio.IsVideo(filename, contentType)
	if st, statErr := os.Stat(srcPath); statErr == nil {
		if isVideo {
			s.metrics.ObserveUploadBytes("video", st.Size())
		} else {
			s.metrics.ObserveUploadBytes("audio", st.Size())
		}
	}
	// inferStart marks the beginning of the inference work (ffmpeg extraction
	// for video, then the upstream call). inference_duration_seconds spans this
	// whole window; upstream_inference_duration_seconds covers only the
	// whisper-server call itself.
	inferStart := time.Now()
	if isVideo {
		s.log.Info("video media detected; extracting audio track", "file", filename)
		s.metrics.IncFFmpegInFlight()
		extractStart := time.Now()
		wavPath, err := audio.Extract(r.Context(), srcPath, s.cfg.Audio.FfmpegBin, s.cfg.Audio.TempDir, s.log)
		if err != nil {
			s.metrics.DecFFmpegInFlight()
			s.metrics.ObserveFFmpegExtraction("error", time.Since(extractStart))
			s.log.Error("audio extraction failed", "error", err)
			writeError(w, http.StatusBadGateway, "server_error",
				fmt.Sprintf("audio extraction from video failed: %v", err), "")
			return
		}
		s.metrics.DecFFmpegInFlight()
		s.metrics.ObserveFFmpegExtraction("success", time.Since(extractStart))
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
			writeError(w, http.StatusInternalServerError, "server_error", "failed to open media file", "")
			return
		}
		defer sf.Close()
		inferFile = sf
	}

	s.log.Info("forwarding inference request",
		"filename", inferName,
		"translate", translate,
		"response_format", params.format,
		"language", params.language,
		"temperature", params.temperature,
	)

	req := whisper.InferRequest{
		File:           inferFile,
		Filename:       inferName,
		Language:       params.language,
		Prompt:         params.prompt,
		Temperature:    params.temperature,
		Translate:      translate,
		ResponseFormat: params.format,
	}

	task := "transcribe"
	if translate {
		task = "translate"
	}
	upstreamStart := time.Now()
	resp, err := s.cli.Infer(r.Context(), req)
	upstreamDur := time.Since(upstreamStart)
	if err != nil {
		s.metrics.ObserveInference(task, "error", time.Since(inferStart))
		s.log.Error("inference failed", "error", err)
		writeError(w, http.StatusBadGateway, "server_error", fmt.Sprintf("upstream inference failed: %v", err), "")
		return
	}
	s.metrics.ObserveInference(task, "success", time.Since(inferStart))
	s.metrics.ObserveUpstreamInference(upstreamDur)
	s.metrics.ObserveTranscribedSeconds(task, resp.Duration)
	s.log.Info("inference completed",
		"filename", inferName,
		"response_format", params.format,
		"text_len", len(resp.Text),
		"raw_len", len(resp.RawBody),
		"duration_ms", time.Since(start).Milliseconds(),
	)
	s.writeInferResponse(w, params.format, translate, resp)
}

// resolveSource resolves the request's media source to a local file path plus
// the per-request inference parameters. It accepts either a multipart form
// (OpenAI-style file upload) or an application/json body with a "url" field
// (downloaded via yt-dlp). On failure it writes the error response and
// returns ok=false.
func (s *Server) resolveSource(w http.ResponseWriter, r *http.Request) (srcPath, filename, contentType string, p inferParams, ok bool) {
	if ct := r.Header.Get("Content-Type"); strings.HasPrefix(strings.ToLower(ct), "application/json") {
		return s.resolveURLSource(w, r)
	}
	return s.resolveMultipartSource(w, r)
}

// resolveMultipartSource spools an OpenAI-style multipart file upload to disk
// and parses the form fields.
func (s *Server) resolveMultipartSource(w http.ResponseWriter, r *http.Request) (srcPath, filename, contentType string, p inferParams, ok bool) {
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
	srcPath, err = audio.Spool(file, header.Filename, s.cfg.Audio.TempDir)
	if err != nil {
		s.log.Error("failed to spool upload", "error", err)
		writeError(w, http.StatusInternalServerError, "server_error", "failed to store the uploaded file", "")
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

	p = inferParams{
		model:       r.FormValue("model"),
		format:      r.FormValue("response_format"),
		language:    r.FormValue("language"),
		prompt:      r.FormValue("prompt"),
		temperature: temperature,
	}
	return srcPath, header.Filename, header.Header.Get("Content-Type"), p, true
}

// resolveURLSource handles an application/json body: it validates the "url"
// field, downloads the media via yt-dlp, and returns the local path.
func (s *Server) resolveURLSource(w http.ResponseWriter, r *http.Request) (srcPath, filename, contentType string, p inferParams, ok bool) {
	var jr jsonInferRequest
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		s.log.Warn("failed to read JSON body", "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error", "failed to read request body", "")
		return
	}
	if err := json.Unmarshal(body, &jr); err != nil {
		s.log.Warn("request body is not valid JSON", "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"request body must be valid JSON with a 'url' field", "")
		return
	}
	if strings.TrimSpace(jr.URL) == "" {
		s.log.Warn("JSON request missing url field")
		writeError(w, http.StatusBadRequest, "invalid_request_error", "no 'url' field in the request", "url")
		return
	}
	if s.dl == nil {
		s.log.Warn("URL ingestion requested but disabled")
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"URL ingestion is disabled (set download.enabled=true)", "url")
		return
	}

	s.log.Info("downloading media from URL", "url", jr.URL)
	s.metrics.IncDownloadInFlight()
	dlStart := time.Now()
	path, filename, err := s.dl.Download(r.Context(), jr.URL)
	if err != nil {
		s.metrics.DecDownloadInFlight()
		s.metrics.ObserveDownload("error", time.Since(dlStart))
		s.log.Error("media download failed", "url", jr.URL, "error", err)
		if errors.Is(err, download.ErrInvalidURL) || errors.Is(err, download.ErrHostNotAllowed) {
			writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error(), "url")
			return
		}
		writeError(w, http.StatusBadGateway, "server_error",
			fmt.Sprintf("failed to download media: %v", err), "")
		return
	}
	s.metrics.DecDownloadInFlight()
	s.metrics.ObserveDownload("success", time.Since(dlStart))
	if st, statErr := os.Stat(path); statErr == nil {
		s.metrics.ObserveDownloadBytes(st.Size())
		s.log.Info("media download complete", "url", jr.URL, "file", filename, "bytes", st.Size())
	} else {
		s.log.Info("media download complete", "url", jr.URL, "file", filename)
	}

	p = inferParams{
		model:       jr.Model,
		format:      jr.ResponseFormat,
		language:    jr.Language,
		prompt:      jr.Prompt,
		temperature: jr.Temperature,
	}
	return path, filename, "", p, true
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
