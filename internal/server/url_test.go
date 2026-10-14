package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tekmanic/whisper-oai/internal/config"
)

// writeFakeYTDLP creates a fake yt-dlp shell script that writes a stub media
// file to the -o template (substituting %(ext)s with ext) and prints the
// final path on stdout, mimicking `--print after_move:filepath`. It records
// its argv to $FAKE_YTDLP_ARGV when set. Returns the script path.
func writeFakeYTDLP(t *testing.T, ext string) string {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "yt-dlp")
	body := `#!/bin/sh
if [ -n "$FAKE_YTDLP_ARGV" ]; then
  printf '%s\n' "$@" > "$FAKE_YTDLP_ARGV"
fi
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-o" ]; then out="$a"; fi
  prev="$a"
done
if [ -z "$out" ]; then
  echo "fake yt-dlp: missing -o template" >&2
  exit 1
fi
out=$(printf '%s' "$out" | sed 's/%(ext)s/` + ext + `/')
printf 'fake media bytes' > "$out"
echo "$out"
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake yt-dlp: %v", err)
	}
	return script
}

// writeFailingYTDLP creates a fake yt-dlp shell script that always exits 1
// with an error on stderr. Returns the script path.
func writeFailingYTDLP(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "yt-dlp")
	body := "#!/bin/sh\necho 'ERROR: [youtube] dQw4w9WgXcQ: Unable to download webpage: HTTP 403' >&2\nexit 1\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write failing yt-dlp: %v", err)
	}
	return script
}

// jsonURLRequest builds a POST request with an application/json body.
func jsonURLRequest(t *testing.T, path, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

// urlTestConfig returns a test config with URL ingestion enabled, pointing at
// the given fake yt-dlp and (optionally) fake ffmpeg.
func urlTestConfig(t *testing.T, ytdlp, ffmpeg string) *config.Config {
	t.Helper()
	cfg := testConfig()
	cfg.Download = config.DownloadConfig{
		Enabled: true,
		Bin:     ytdlp,
		Timeout: time.Minute,
		Format:  "bv*+ba/b",
	}
	if ffmpeg != "" {
		cfg.Audio = config.AudioConfig{FfmpegBin: ffmpeg}
	}
	return cfg
}

// TestURLTranscriptionVideo verifies that a JSON url request downloads the
// media via (fake) yt-dlp, extracts the audio track via (fake) ffmpeg, and
// forwards the resulting .wav to whisper-server.
func TestURLTranscriptionVideo(t *testing.T) {
	fake, fw := newFakeWhisper(t)

	ytdlp := writeFakeYTDLP(t, "mp4")
	ffmpeg := writeFakeFFmpeg(t)
	argvFile := filepath.Join(t.TempDir(), "argv.txt")
	t.Setenv("FAKE_FFMPEG_ARGV", argvFile)

	s := newTestServer(t, urlTestConfig(t, ytdlp, ffmpeg), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","model":"whisper-1","response_format":"json"}`)
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
	// not the downloaded mp4.
	_, filename, _ := fw.snapshot()
	if !strings.HasSuffix(filename, ".wav") {
		t.Errorf("upstream filename = %q, want *.wav", filename)
	}

	// The fake ffmpeg must have been invoked on the downloaded file.
	argvRaw, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("fake ffmpeg was not invoked: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(argvRaw)), "\n")
	if !strings.Contains(got[6], "whisper-oai-dl-") {
		t.Errorf("ffmpeg argv[6] = %q, want downloaded file path", got[6])
	}
}

// TestURLTranscriptionAudio verifies that a URL download producing a plain
// audio file (m4a) skips ffmpeg entirely and is forwarded as-is.
func TestURLTranscriptionAudio(t *testing.T) {
	fake, fw := newFakeWhisper(t)

	ytdlp := writeFakeYTDLP(t, "m4a")
	ffmpeg := writeFakeFFmpeg(t)
	argvFile := filepath.Join(t.TempDir(), "argv.txt")
	t.Setenv("FAKE_FFMPEG_ARGV", argvFile)

	s := newTestServer(t, urlTestConfig(t, ytdlp, ffmpeg), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://example.com/podcast.mp3","model":"whisper-1"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	_, filename, _ := fw.snapshot()
	if !strings.HasSuffix(filename, ".m4a") {
		t.Errorf("upstream filename = %q, want *.m4a (ffmpeg must be skipped)", filename)
	}
	if _, err := os.Stat(argvFile); !os.IsNotExist(err) {
		t.Errorf("fake ffmpeg was invoked for an audio-only download")
	}
}

// TestURLTranscriptionVerboseJSON verifies that response_format and language
// from the JSON body are forwarded to whisper-server.
func TestURLTranscriptionVerboseJSON(t *testing.T) {
	fake, fw := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "wav")

	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/translations",
		`{"url":"https://example.com/a.wav","model":"whisper-1","response_format":"verbose_json","language":"en"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var vj struct {
		Task     string `json:"task"`
		Language string `json:"language"`
		Text     string `json:"text"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &vj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if vj.Text != cannedText {
		t.Errorf("text = %q, want %q", vj.Text, cannedText)
	}
	fields, _, _ := fw.snapshot()
	if fields["response_format"] != "verbose_json" {
		t.Errorf("upstream response_format = %q, want verbose_json", fields["response_format"])
	}
	if fields["language"] != "en" {
		t.Errorf("upstream language = %q, want en", fields["language"])
	}
	if fields["translate"] != "true" {
		t.Errorf("upstream translate = %q, want true", fields["translate"])
	}
}

// TestURLDisabled verifies that a JSON url request gets 400 when URL
// ingestion is disabled (the default test config).
func TestURLDisabled(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	s := newTestServer(t, testConfig(), fake.URL) // Download zero value → disabled

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://example.com/a.mp4","model":"whisper-1"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj == nil || !strings.Contains(errObj["message"].(string), "disabled") {
		t.Errorf("error = %v, want disabled message", errObj)
	}
}

// TestURLMissing verifies that a JSON body without a url field gets 400 with
// param "url".
func TestURLMissing(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "mp4")
	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions", `{"model":"whisper-1"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj == nil || errObj["param"] != "url" {
		t.Errorf("error = %v, want param url", errObj)
	}
}

// TestURLInvalidJSON verifies that a malformed JSON body gets 400.
func TestURLInvalidJSON(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "mp4")
	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions", `{not json`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
}

// TestURLInvalidScheme verifies that non-http(s) URLs are rejected with 400
// (param "url") before any download is attempted.
func TestURLInvalidScheme(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "mp4")
	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	for _, url := range []string{"file:///etc/passwd", "ftp://example.com/a.mp4", "notaurl"} {
		rec := httptest.NewRecorder()
		req := jsonURLRequest(t, "/v1/audio/transcriptions",
			`{"url":"`+url+`","model":"whisper-1"}`)
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("url %q: status = %d, want 400: %s", url, rec.Code, rec.Body.String())
		}
	}
}

// TestURLHostNotAllowed verifies that the host allowlist rejects disallowed
// hosts with 400.
func TestURLHostNotAllowed(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "mp4")

	cfg := urlTestConfig(t, ytdlp, "")
	cfg.Download.AllowedHosts = []string{"example.com", "*.youtube.com"}
	s := newTestServer(t, cfg, fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://evil.example.net/a.mp4","model":"whisper-1"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj == nil || errObj["param"] != "url" {
		t.Errorf("error = %v, want param url", errObj)
	}
}

// TestURLDownloadFailure verifies that a failing yt-dlp produces a 502 with
// the OpenAI error shape and the yt-dlp stderr in the message.
func TestURLDownloadFailure(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFailingYTDLP(t)

	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","model":"whisper-1"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj == nil || !strings.Contains(errObj["message"].(string), "Unable to download webpage") {
		t.Errorf("error = %v, want yt-dlp stderr included", errObj)
	}
}

// TestURLWrongModel verifies that model validation still applies to JSON url
// requests.
func TestURLWrongModel(t *testing.T) {
	fake, _ := newFakeWhisper(t)
	ytdlp := writeFakeYTDLP(t, "mp4")
	s := newTestServer(t, urlTestConfig(t, ytdlp, ""), fake.URL)

	rec := httptest.NewRecorder()
	req := jsonURLRequest(t, "/v1/audio/transcriptions",
		`{"url":"https://example.com/a.mp4","model":"gpt-4o-transcribe"}`)
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj == nil || errObj["param"] != "model" {
		t.Errorf("error = %v, want param model", errObj)
	}
}
