//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

func TestOpenWebUIOllamaToolCallsWhisperAndFormatsResult(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("integration test skipped in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Minute)
	defer cancel()

	repoRoot := findRepoRoot(t)
	audioPath := filepath.Join(repoRoot, "testdata", "audio.wav")
	assertFixture(t, audioPath)

	whisperBaseURL := startWhisperStackForLLMTest(t, ctx, repoRoot)
	openWebUIBaseURL, ollamaModel, openWebUIToken := startOpenWebUIWithOllama(t, ctx)

	chatURL := openWebUIBaseURL + "/ollama/api/chat"
	messages := []ollamaMessage{
		{Role: "system", Content: "You are a helpful assistant. Always call transcribe_fixture exactly once before answering."},
		{Role: "user", Content: "Use transcribe_fixture on fixture testdata/audio.wav and then return a short, nicely formatted markdown summary."},
	}

	tools := []ollamaTool{
		{
			Type: "function",
			Function: ollamaToolFn{
				Name:        "transcribe_fixture",
				Description: "Transcribe an audio fixture by calling whisper-oai.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"fixture": map[string]any{
							"type":        "string",
							"description": "Repository-relative fixture path, e.g. testdata/audio.wav",
						},
					},
					"required": []string{"fixture"},
				},
			},
		},
	}

	firstResp := callOllamaChat(t, ctx, chatURL, openWebUIToken, ollamaChatRequest{
		Model:    ollamaModel,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	})
	if len(firstResp.Message.ToolCalls) == 0 {
		t.Fatalf("expected at least one tool call; got response: %q", firstResp.Message.Content)
	}

	toolCall := firstResp.Message.ToolCalls[0]
	if toolCall.Function.Name != "transcribe_fixture" {
		t.Fatalf("unexpected tool call name: %q", toolCall.Function.Name)
	}

	fixtureArg := parseFixtureArg(t, toolCall.Function.Arguments)
	fixtureAbsPath := filepath.Join(repoRoot, filepath.Clean(fixtureArg))
	if !strings.HasPrefix(fixtureAbsPath, filepath.Join(repoRoot, "testdata")+string(filepath.Separator)) {
		t.Fatalf("tool requested fixture outside testdata: %q", fixtureArg)
	}

	transcribed := transcribeFile(t, ctx, whisperBaseURL, fixtureAbsPath)
	if transcribed == "" {
		t.Fatal("expected non-empty transcription from whisper-oai")
	}
	transcribedForModel := prepareTranscriptForModel(transcribed)

	messages = append(messages, firstResp.Message)
	messages = append(messages, ollamaMessage{
		Role:    "tool",
		Name:    "transcribe_fixture",
		Content: transcribedForModel,
	})

	finalResp := callOllamaChat(t, ctx, chatURL, openWebUIToken, ollamaChatRequest{
		Model:    ollamaModel,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	})

	formatted := strings.TrimSpace(finalResp.Message.Content)
	if formatted == "" {
		t.Fatal("expected non-empty formatted response from Open WebUI/Ollama")
	}
	if !strings.Contains(formatted, "-") && !strings.Contains(formatted, "#") {
		t.Fatalf("expected structured summary output, got: %q", formatted)
	}
}

func prepareTranscriptForModel(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	lines := strings.Split(trimmed, "\n")
	kept := make([]string, 0, len(lines))
	for _, line := range lines {
		l := strings.TrimSpace(line)
		if l == "" || strings.EqualFold(l, "[Music]") {
			continue
		}
		kept = append(kept, l)
	}

	cleaned := strings.Join(kept, "\n")
	if cleaned == "" {
		cleaned = trimmed
	}

	const maxChars = 1800
	if len(cleaned) > maxChars {
		cleaned = cleaned[:maxChars]
	}

	return cleaned
}

func startWhisperStackForLLMTest(t *testing.T, ctx context.Context, repoRoot string) string {
	t.Helper()

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
	t.Cleanup(func() { _ = whisperCpp.Terminate(context.Background()) })

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
				"WHISPER_OAI_AUDIO_FFMPEG_BIN":   "/bin/false",
				"WHISPER_OAI_SERVER_MODEL_NAME":  "whisper-1",
			},
			HostConfigModifier: func(hc *dockercontainer.HostConfig) {
				hc.ExtraHosts = append(hc.ExtraHosts, "host.docker.internal:host-gateway")
			},
			WaitingFor: wait.ForHTTP("/healthz").WithPort("8000/tcp").WithStartupTimeout(4 * time.Minute),
		},
		Started: true,
	})
	if err != nil {
		t.Fatalf("start whisper-oai container: %v", err)
	}
	t.Cleanup(func() { _ = whisperOAI.Terminate(context.Background()) })

	proxyHost, err := whisperOAI.Host(ctx)
	if err != nil {
		t.Fatalf("whisper-oai host: %v", err)
	}
	proxyPort, err := whisperOAI.MappedPort(ctx, "8000/tcp")
	if err != nil {
		t.Fatalf("whisper-oai mapped port: %v", err)
	}

	return fmt.Sprintf("http://%s:%s", proxyHost, proxyPort.Port())
}

func startOpenWebUIWithOllama(t *testing.T, ctx context.Context) (openWebUIBaseURL string, model string, authToken string) {
	t.Helper()

	const ollamaModel = "qwen2.5:0.5b"

	imageCandidates := []string{
		"ghcr.io/open-webui/open-webui:ollama",
		"openwebui/open-webui:ollama",
		"open-webui/open-webui:ollama",
	}

	if override := strings.TrimSpace(getenv("OPENWEBUI_IMAGE", "")); override != "" {
		imageCandidates = []string{override}
	}

	var (
		openwebui testcontainers.Container
		err       error
	)
	startErrors := make([]string, 0, len(imageCandidates))
	for _, image := range imageCandidates {
		openwebui, err = testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
			ContainerRequest: testcontainers.ContainerRequest{
				Image:        image,
				ExposedPorts: []string{"8080/tcp"},
				Env: map[string]string{
					"WEBUI_AUTH":    "true",
					"ENABLE_SIGNUP": "true",
				},
				HostConfigModifier: func(hc *dockercontainer.HostConfig) {
					hc.ExtraHosts = append(hc.ExtraHosts, "host.docker.internal:host-gateway")
				},
				WaitingFor: wait.ForHTTP("/").WithPort("8080/tcp").WithStartupTimeout(6 * time.Minute),
			},
			Started: true,
		})
		if err == nil {
			break
		}
		startErrors = append(startErrors, fmt.Sprintf("%s => %v", image, err))
	}
	if err != nil {
		t.Fatalf("start openwebui container failed across candidate images: %s", strings.Join(startErrors, " | "))
	}
	t.Cleanup(func() { _ = openwebui.Terminate(context.Background()) })

	// Use the bundled ollama runtime inside the Open WebUI container.
	if _, _, err = openwebui.Exec(ctx, []string{"ollama", "pull", ollamaModel}); err != nil {
		t.Fatalf("pull ollama model %q in openwebui container: %v", ollamaModel, err)
	}

	webHost, err := openwebui.Host(ctx)
	if err != nil {
		t.Fatalf("openwebui host: %v", err)
	}
	webPort, err := openwebui.MappedPort(ctx, "8080/tcp")
	if err != nil {
		t.Fatalf("openwebui mapped port: %v", err)
	}
	baseURL := fmt.Sprintf("http://%s:%s", webHost, webPort.Port())
	token := bootstrapOpenWebUIAuth(t, ctx, baseURL)

	return baseURL, ollamaModel, token
}

func getenv(key, fallback string) string {
	raw, ok := os.LookupEnv(key)
	v := strings.TrimSpace(raw)
	if !ok {
		v = ""
	}
	if v == "" {
		return fallback
	}
	return v
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Tools    []ollamaTool    `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
}

type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content,omitempty"`
	Name      string           `json:"name,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaTool struct {
	Type     string       `json:"type"`
	Function ollamaToolFn `json:"function"`
}

type ollamaToolFn struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type ollamaToolCall struct {
	Function ollamaToolCallFn `json:"function"`
}

type ollamaToolCallFn struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func callOllamaChat(t *testing.T, ctx context.Context, chatURL, authToken string, req ollamaChatRequest) ollamaChatResponse {
	t.Helper()

	payload, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal ollama chat request: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, chatURL, bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("build ollama chat request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(authToken) != "" {
		httpReq.Header.Set("Authorization", "Bearer "+authToken)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatalf("send ollama chat request: %v", err)
	}
	defer resp.Body.Close()

	body := new(bytes.Buffer)
	if _, err := body.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read ollama chat response: %v", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		t.Fatalf("ollama chat request failed: status=%d body=%s", resp.StatusCode, body.String())
	}

	var out ollamaChatResponse
	if err := json.Unmarshal(body.Bytes(), &out); err != nil {
		t.Fatalf("decode ollama chat response: %v (body=%s)", err, body.String())
	}
	return out
}

func bootstrapOpenWebUIAuth(t *testing.T, ctx context.Context, baseURL string) string {
	t.Helper()

	credentials := map[string]string{
		"name":     "integration-test",
		"email":    "integration@example.com",
		"password": "integration-test-password",
	}

	signinPayload := map[string]string{
		"email":    credentials["email"],
		"password": credentials["password"],
	}

	if token, ok := tryExtractSigninToken(t, ctx, baseURL+"/api/v1/auths/signin", signinPayload); ok {
		return token
	}

	// First-account signup path (or tolerated if account already exists / backend is mid-init).
	if signupBody, signupStatus := postJSONWithStatus(t, ctx, baseURL+"/api/v1/auths/signup", credentials); signupStatus < 200 || signupStatus >= 300 {
		text := strings.ToLower(string(signupBody))
		if !strings.Contains(text, "already") && !strings.Contains(text, "exists") && !strings.Contains(text, "internal error") {
			t.Fatalf("openwebui signup failed: status=%d body=%s", signupStatus, string(signupBody))
		}
	}

	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) {
		if token, ok := tryExtractSigninToken(t, ctx, baseURL+"/api/v1/auths/signin", signinPayload); ok {
			return token
		}
		time.Sleep(2 * time.Second)
	}

	t.Fatal("unable to obtain Open WebUI auth token after signup/signin retries")
	return ""
}

func tryExtractSigninToken(t *testing.T, ctx context.Context, signinURL string, payload map[string]string) (string, bool) {
	t.Helper()

	body, status := postJSONWithStatus(t, ctx, signinURL, payload)
	if status < 200 || status >= 300 {
		return "", false
	}

	var parsed struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", false
	}

	token := strings.TrimSpace(parsed.Token)
	if token == "" {
		return "", false
	}

	return token, true
}

func postJSON(t *testing.T, ctx context.Context, url string, body any) []byte {
	t.Helper()

	respBody, status := postJSONWithStatus(t, ctx, url, body)
	if status < 200 || status >= 300 {
		t.Fatalf("POST %s failed: status=%d body=%s", url, status, string(respBody))
	}

	return respBody
}

func postJSONWithStatus(t *testing.T, ctx context.Context, url string, body any) ([]byte, int) {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request body for %s: %v", url, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("build POST request %s: %v", url, err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("send POST request %s: %v", url, err)
	}
	defer resp.Body.Close()

	respBody := new(bytes.Buffer)
	if _, err := respBody.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read POST response %s: %v", url, err)
	}

	return respBody.Bytes(), resp.StatusCode
}

func parseFixtureArg(t *testing.T, raw json.RawMessage) string {
	t.Helper()

	var obj struct {
		Fixture string `json:"fixture"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Fixture != "" {
		return obj.Fixture
	}

	// Some models emit arguments as a JSON-encoded string.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil && s != "" {
		if err := json.Unmarshal([]byte(s), &obj); err == nil && obj.Fixture != "" {
			return obj.Fixture
		}
	}

	t.Fatalf("tool call arguments did not contain fixture: %s", string(raw))
	return ""
}
