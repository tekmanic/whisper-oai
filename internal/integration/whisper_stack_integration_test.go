//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

func TestAudioAndVideoProduceSameTranscription(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("integration test skipped in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	repoRoot := findRepoRoot(t)
	audioPath := filepath.Join(repoRoot, "testdata", "audio.wav")
	videoPath := filepath.Join(repoRoot, "testdata", "video.mp4")
	assertFixture(t, audioPath)
	assertFixture(t, videoPath)

	whisperCpp, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: testcontainers.ContainerRequest{
			FromDockerfile: testcontainers.FromDockerfile{
				Context:    repoRoot,
				Dockerfile: "testdata/integration/Dockerfile.whispercpp",
			},
			ExposedPorts: []string{"8080/tcp"},
			WaitingFor:   wait.ForHTTP("/health").WithPort("8080/tcp").WithStartupTimeout(5 * time.Minute),
		},
		Started: true,
	})
	if err != nil {
		t.Fatalf("start whisper.cpp container: %v", err)
	}
	t.Cleanup(func() {
		_ = whisperCpp.Terminate(context.Background())
	})

	whisperCppHost, err := whisperCpp.Host(ctx)
	if err != nil {
		t.Fatalf("whisper.cpp host: %v", err)
	}
	whisperCppPort, err := whisperCpp.MappedPort(ctx, "8080/tcp")
	if err != nil {
		t.Fatalf("whisper.cpp mapped port: %v", err)
	}
	whisperRemoteURL := fmt.Sprintf("http://host.docker.internal:%s", whisperCppPort.Port())

	whisperOAI, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: testcontainers.ContainerRequest{
			FromDockerfile: testcontainers.FromDockerfile{
				Context:    repoRoot,
				Dockerfile: "testdata/integration/Dockerfile.whisper-oai",
			},
			ExposedPorts: []string{"8000/tcp"},
			Env: map[string]string{
				"WHISPER_OAI_WHISPER_REMOTE_URL": whisperRemoteURL,
				"WHISPER_OAI_AUDIO_FFMPEG_BIN":   "ffmpeg",
				"WHISPER_OAI_SERVER_MODEL_NAME":  "whisper-1",
			},
			HostConfigModifier: func(hc *dockercontainer.HostConfig) {
				hc.ExtraHosts = append(hc.ExtraHosts, "host.docker.internal:host-gateway")
			},
			WaitingFor: wait.ForHTTP("/healthz").WithPort("8000/tcp").WithStartupTimeout(3 * time.Minute),
		},
		Started: true,
	})
	if err != nil {
		t.Fatalf("start whisper-oai container: %v", err)
	}
	t.Cleanup(func() {
		_ = whisperOAI.Terminate(context.Background())
	})

	proxyHost, err := whisperOAI.Host(ctx)
	if err != nil {
		t.Fatalf("whisper-oai host: %v", err)
	}
	proxyPort, err := whisperOAI.MappedPort(ctx, "8000/tcp")
	if err != nil {
		t.Fatalf("whisper-oai mapped port: %v", err)
	}
	baseURL := fmt.Sprintf("http://%s:%s", proxyHost, proxyPort.Port())

	audioText := transcribeFile(t, ctx, baseURL, audioPath)
	videoText := transcribeFile(t, ctx, baseURL, videoPath)

	audioNorm := normalizeTranscription(audioText)
	videoNorm := normalizeTranscription(videoText)
	audioWords := normalizeWords(audioNorm)
	videoWords := normalizeWords(videoNorm)
	editSimilarity := wordSimilarity(audioWords, videoWords)
	containment := orderedContainmentSimilarity(audioWords, videoWords)
	similarity := containment
	if editSimilarity > similarity {
		similarity = editSimilarity
	}

	// ASR from audio.wav and extracted video audio can differ in alignment and length,
	// but they should remain semantically close at the token sequence level.
	const minSimilarity = 0.60
	if similarity < minSimilarity {
		t.Logf("transcriptions differ after normalization: similarity=%.3f", similarity)
		t.Logf("edit similarity=%.3f containment similarity=%.3f", editSimilarity, containment)
		t.Logf("audio words=%d video words=%d", len(audioWords), len(videoWords))
		t.Logf("word-level diff excerpt:\n%s", unifiedDiff("audio.wav", "video.mp4", strings.Join(audioWords, "\n"), strings.Join(videoWords, "\n")))
		t.Fatalf("transcriptions differ after normalization and alignment")
	}
	if len(audioWords) == 0 {
		t.Fatalf("expected non-empty transcription, got empty string")
	}

	t.Logf("comparison complete using whisper.cpp on Debian trixie: backend=%s:%s", whisperCppHost, whisperCppPort.Port())
}

func transcribeFile(t *testing.T, ctx context.Context, baseURL, filePath string) string {
	t.Helper()

	f, err := os.Open(filePath)
	if err != nil {
		t.Fatalf("open file %s: %v", filePath, err)
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := io.Copy(fw, f); err != nil {
		t.Fatalf("copy file: %v", err)
	}
	if err := mw.WriteField("model", "whisper-1"); err != nil {
		t.Fatalf("write model field: %v", err)
	}
	if err := mw.WriteField("response_format", "json"); err != nil {
		t.Fatalf("write response_format field: %v", err)
	}
	if err := mw.WriteField("temperature", "0"); err != nil {
		t.Fatalf("write temperature field: %v", err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/audio/transcriptions", &body)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("send request: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("transcription request failed: status=%d body=%s", resp.StatusCode, string(respBody))
	}

	var payload struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(respBody, &payload); err != nil {
		t.Fatalf("decode transcription response: %v (body=%s)", err, string(respBody))
	}
	return payload.Text
}

func assertFixture(t *testing.T, path string) {
	t.Helper()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("fixture does not exist: %s (%v)", path, err)
	}
	if st.Size() == 0 {
		t.Fatalf("fixture is empty: %s", path)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(currentFile)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("go.mod not found from %s", currentFile)
		}
		dir = parent
	}
}

func normalizeTranscription(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}

	// Remove common ASR filler tags that are semantically meaningless for this test.
	s = fillerTagRE.ReplaceAllString(s, " ")

	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, strings.Join(strings.Fields(line), " "))
	}

	return strings.Join(out, "\n")
}

var fillerTagRE = regexp.MustCompile(`(?i)\[(?:blank\s+audio|silence|noise|music|inaudible|unintelligible|applause)\]|<(?:blank\s+audio|silence|noise|music|inaudible|unintelligible|applause)>`)

func normalizeWords(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}

	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f == "" {
			continue
		}
		if f == "uh" || f == "um" {
			continue
		}
		out = append(out, f)
	}
	out = removeBlankAudioPairs(out)
	out = squashRepeatedWindows(out, 5, 3)
	return out
}

func removeBlankAudioPairs(words []string) []string {
	if len(words) < 2 {
		return words
	}
	out := make([]string, 0, len(words))
	for i := 0; i < len(words); i++ {
		if i+1 < len(words) && words[i] == "blank" && words[i+1] == "audio" {
			i++
			continue
		}
		out = append(out, words[i])
	}
	return out
}

func squashRepeatedWindows(words []string, window, minRepeats int) []string {
	if len(words) < window*minRepeats || window <= 0 || minRepeats <= 1 {
		return words
	}

	out := make([]string, 0, len(words))
	for i := 0; i < len(words); {
		if i+window*minRepeats <= len(words) {
			base := strings.Join(words[i:i+window], "\x00")
			repeats := 1
			for j := i + window; j+window <= len(words); j += window {
				if strings.Join(words[j:j+window], "\x00") != base {
					break
				}
				repeats++
			}
			if repeats >= minRepeats {
				out = append(out, words[i:i+window]...)
				i += repeats * window
				continue
			}
		}

		out = append(out, words[i])
		i++
	}

	return out
}

func wordSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	d := levenshteinWords(a, b)
	den := len(a)
	if len(b) > den {
		den = len(b)
	}
	return 1 - float64(d)/float64(den)
}

func orderedContainmentSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	shorter := a
	longer := b
	if len(a) > len(b) {
		shorter = b
		longer = a
	}

	i, j, matched := 0, 0, 0
	for i < len(shorter) && j < len(longer) {
		if shorter[i] == longer[j] {
			matched++
			i++
			j++
			continue
		}
		j++
	}

	return float64(matched) / float64(len(shorter))
}

func levenshteinWords(a, b []string) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}

	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := 0; j <= len(b); j++ {
		prev[j] = j
	}

	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := cur[j-1] + 1
			sub := prev[j-1] + cost
			cur[j] = min3(del, ins, sub)
		}
		prev, cur = cur, prev
	}

	return prev[len(b)]
}

func min3(a, b, c int) int {
	if a <= b && a <= c {
		return a
	}
	if b <= a && b <= c {
		return b
	}
	return c
}

func unifiedDiff(from, to, a, b string) string {
	aLines := strings.Split(a, "\n")
	bLines := strings.Split(b, "\n")
	max := len(aLines)
	if len(bLines) > max {
		max = len(bLines)
	}

	var sb strings.Builder
	sb.WriteString("--- " + from + "\n")
	sb.WriteString("+++ " + to + "\n")

	for i := 0; i < max; i++ {
		var av, bv string
		if i < len(aLines) {
			av = aLines[i]
		}
		if i < len(bLines) {
			bv = bLines[i]
		}
		if av == bv {
			continue
		}
		if i < len(aLines) {
			sb.WriteString("- " + av + "\n")
		}
		if i < len(bLines) {
			sb.WriteString("+ " + bv + "\n")
		}
	}

	return sb.String()
}
