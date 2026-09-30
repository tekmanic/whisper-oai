package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tekmanic/whisper-oai/internal/config"
)

// writeFakeFFmpeg creates a fake ffmpeg shell script that records its argv to
// $FAKE_FFMPEG_ARGV and writes a stub WAV to its last argument (the output
// path). Returns the script path.
func writeFakeFFmpeg(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "ffmpeg")
	body := `#!/bin/sh
if [ -n "$FAKE_FFMPEG_ARGV" ]; then
  printf '%s\n' "$@" > "$FAKE_FFMPEG_ARGV"
fi
for last; do :; done
printf 'RIFF-fake-wav' > "$last"
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake ffmpeg: %v", err)
	}
	return script
}

// TestVideoUpload verifies that a video upload (mp4) is spooled, its audio
// track "extracted" via (fake) ffmpeg, and the resulting .wav is what gets
// forwarded to whisper-server.
func TestVideoUpload(t *testing.T) {
	fake, fw := newFakeWhisper(t)

	script := writeFakeFFmpeg(t)
	argvFile := filepath.Join(t.TempDir(), "argv.txt")
	t.Setenv("FAKE_FFMPEG_ARGV", argvFile)

	cfg := testConfig()
	cfg.Audio = config.AudioConfig{FfmpegBin: script}
	s := newTestServer(t, cfg, fake.URL)

	rec := httptest.NewRecorder()
	req := multipartRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "whisper-1", "response_format": "json"},
		"movie.mp4", []byte("fake mp4 bytes"))
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body["text"] != cannedText {
		t.Errorf("text = %q, want %q", body["text"], cannedText)
	}

	// The fake whisper-server must have received a .wav (the extracted audio),
	// not the original mp4.
	_, filename, _ := fw.snapshot()
	if !strings.HasSuffix(filename, ".wav") {
		t.Errorf("upstream filename = %q, want *.wav", filename)
	}

	// The fake ffmpeg must have been invoked with the exact expected argv.
	argvRaw, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("fake ffmpeg was not invoked: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(argvRaw)), "\n")
	wantLen := 15 // -nostdin -hide_banner -loglevel error -y -i <src> -vn -ac 1 -ar 16000 -c:a pcm_s16le <dst>
	if len(got) != wantLen {
		t.Fatalf("ffmpeg argv = %v (len %d), want len %d", got, len(got), wantLen)
	}
	prefix := []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"}
	for i, want := range prefix {
		if got[i] != want {
			t.Errorf("argv[%d] = %q, want %q", i, got[i], want)
		}
	}
	if !strings.Contains(got[6], "whisper-oai-upload-") {
		t.Errorf("argv[6] = %q, want spooled upload path", got[6])
	}
	middle := []string{"-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"}
	for i, want := range middle {
		if got[7+i] != want {
			t.Errorf("argv[%d] = %q, want %q", 7+i, got[7+i], want)
		}
	}
	if !strings.HasSuffix(got[14], ".wav") {
		t.Errorf("argv[14] = %q, want *.wav output path", got[14])
	}
}

// TestVideoUploadExtractionFailure verifies that a failing ffmpeg produces a
// 502 with the OpenAI error shape.
func TestVideoUploadExtractionFailure(t *testing.T) {
	fake, _ := newFakeWhisper(t)

	dir := t.TempDir()
	script := filepath.Join(dir, "ffmpeg")
	body := "#!/bin/sh\necho 'ffmpeg: Invalid data found when processing input' >&2\nexit 1\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake ffmpeg: %v", err)
	}

	cfg := testConfig()
	cfg.Audio = config.AudioConfig{FfmpegBin: script}
	s := newTestServer(t, cfg, fake.URL)

	rec := httptest.NewRecorder()
	req := multipartRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "whisper-1"},
		"movie.mp4", []byte("fake mp4 bytes"))
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, ok := raw["error"].(map[string]any)
	if !ok {
		t.Fatalf("no error object: %s", rec.Body.String())
	}
	if !strings.Contains(errObj["message"].(string), "Invalid data found") {
		t.Errorf("message = %v, want ffmpeg stderr included", errObj["message"])
	}
}

// TestUploadTooLarge verifies that uploads exceeding audio.max_upload_mb get
// HTTP 413.
func TestUploadTooLarge(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	cfg := testConfig()
	cfg.Audio = config.AudioConfig{MaxUploadMB: 1}
	s := newTestServer(t, cfg, fake.URL)

	rec := httptest.NewRecorder()
	req := multipartRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "whisper-1"},
		"big.wav", bytes.Repeat([]byte("x"), 2<<20)) // 2 MiB > 1 MiB limit
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413: %s", rec.Code, rec.Body.String())
	}
}
