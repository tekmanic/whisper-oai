package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// clearFlags resets the package-level flag overrides between tests.
func clearFlags() {
	flagMu.Lock()
	defer flagMu.Unlock()
	flagOverrides = map[string]string{}
}

func writeConfigFile(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "whisper-oai.yaml")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write config file: %v", err)
	}
	return p
}

func TestLoadDefaults(t *testing.T) {
	clearFlags()
	c, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := Config{
		Server: ServerConfig{
			Host:      "0.0.0.0",
			Port:      8000,
			APIKey:    "",
			ModelName: "whisper-1",
		},
		Whisper: WhisperConfig{
			Bin:       "/opt/whisper.cpp/build/bin/whisper-server",
			Model:     "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin",
			RemoteURL: "",
			Device:    0,
			FlashAttn: true,
			Host:      "127.0.0.1",
			Port:      8080,
			Threads:   0,
			Language:  "auto",
			ExtraArgs: nil,
		},
		Audio: AudioConfig{
			FfmpegBin:   "ffmpeg",
			TempDir:     "",
			MaxUploadMB: 2048,
		},
		Metrics: MetricsConfig{
			Enabled:        true,
			HealthInterval: 15 * time.Second,
		},
		Log: LogConfig{Level: "info"},
	}
	if !reflect.DeepEqual(*c, want) {
		t.Errorf("Load() = %+v, want %+v", *c, want)
	}
}

func TestLoadFromFile(t *testing.T) {
	clearFlags()
	path := writeConfigFile(t, `
server:
  host: "10.0.0.1"
  port: 9000
whisper:
  bin: "/usr/local/bin/whisper-server"
  threads: 8
  extra_args:
    - "-t"
    - "4"
log:
  level: "debug"
`)
	c, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Server.Host != "10.0.0.1" {
		t.Errorf("Server.Host = %q, want 10.0.0.1", c.Server.Host)
	}
	if c.Server.Port != 9000 {
		t.Errorf("Server.Port = %d, want 9000", c.Server.Port)
	}
	if c.Server.ModelName != "whisper-1" {
		t.Errorf("Server.ModelName = %q, want default whisper-1", c.Server.ModelName)
	}
	if c.Whisper.Bin != "/usr/local/bin/whisper-server" {
		t.Errorf("Whisper.Bin = %q", c.Whisper.Bin)
	}
	if c.Whisper.Model != "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin" {
		t.Errorf("Whisper.Model = %q, want default", c.Whisper.Model)
	}
	if c.Whisper.Threads != 8 {
		t.Errorf("Whisper.Threads = %d, want 8", c.Whisper.Threads)
	}
	if !c.Whisper.FlashAttn {
		t.Error("Whisper.FlashAttn = false, want default true")
	}
	if !reflect.DeepEqual(c.Whisper.ExtraArgs, []string{"-t", "4"}) {
		t.Errorf("Whisper.ExtraArgs = %v, want [-t 4]", c.Whisper.ExtraArgs)
	}
	if c.Log.Level != "debug" {
		t.Errorf("Log.Level = %q, want debug", c.Log.Level)
	}
	if c.Audio.MaxUploadMB != 2048 {
		t.Errorf("Audio.MaxUploadMB = %d, want default 2048", c.Audio.MaxUploadMB)
	}
}

func TestLoadMissingFile(t *testing.T) {
	clearFlags()
	c, err := Load(filepath.Join(t.TempDir(), "does-not-exist.yaml"))
	if err != nil {
		t.Fatalf("Load with missing file: %v", err)
	}
	if c.Server.Port != 8000 {
		t.Errorf("Server.Port = %d, want default 8000", c.Server.Port)
	}
}

func TestLoadInvalidFile(t *testing.T) {
	clearFlags()
	path := writeConfigFile(t, "server: [unclosed")
	if _, err := Load(path); err == nil {
		t.Fatal("Load with invalid yaml: expected error, got nil")
	}
}

func TestLoadEnvOverridesFile(t *testing.T) {
	clearFlags()
	t.Setenv("WHISPER_OAI_SERVER_PORT", "9999")
	t.Setenv("WHISPER_OAI_WHISPER_BIN", "/env/bin/whisper-server")
	path := writeConfigFile(t, "server:\n  port: 8100\n")
	c, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Server.Port != 9999 {
		t.Errorf("Server.Port = %d, want 9999 (env over file)", c.Server.Port)
	}
	if c.Whisper.Bin != "/env/bin/whisper-server" {
		t.Errorf("Whisper.Bin = %q, want env value", c.Whisper.Bin)
	}
}

func TestLoadFlagOverridesEnv(t *testing.T) {
	clearFlags()
	t.Setenv("WHISPER_OAI_SERVER_PORT", "9999")
	SetFlag("server.port", "7777")
	SetFlag("whisper.flash_attn", "false")
	c, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Server.Port != 7777 {
		t.Errorf("Server.Port = %d, want 7777 (flag over env)", c.Server.Port)
	}
	if c.Whisper.FlashAttn {
		t.Error("Whisper.FlashAttn = true, want false (flag over default)")
	}
}

func TestLoadFlagOverridesDefault(t *testing.T) {
	clearFlags()
	SetFlag("server.host", "127.0.0.1")
	SetFlag("whisper.port", "9090")
	SetFlag("server.model_name", "my-model")
	c, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Server.Host != "127.0.0.1" {
		t.Errorf("Server.Host = %q, want 127.0.0.1", c.Server.Host)
	}
	if c.Whisper.Port != 9090 {
		t.Errorf("Whisper.Port = %d, want 9090", c.Whisper.Port)
	}
	if c.Server.ModelName != "my-model" {
		t.Errorf("Server.ModelName = %q, want my-model", c.Server.ModelName)
	}
}

func TestValidate(t *testing.T) {
	base := func(t *testing.T) *Config {
		t.Helper()
		clearFlags()
		c, err := Load("")
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		return c
	}
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{"valid defaults", func(c *Config) {}, false},
		{"empty bin", func(c *Config) { c.Whisper.Bin = "" }, true},
		{"empty model", func(c *Config) { c.Whisper.Model = "" }, true},
		{"remote mode allows empty local process fields", func(c *Config) {
			c.Whisper.RemoteURL = "http://remote-whisper.internal:8080"
			c.Whisper.Bin = ""
			c.Whisper.Model = ""
			c.Whisper.Host = ""
			c.Whisper.Port = 0
		}, false},
		{"remote mode rejects invalid URL", func(c *Config) {
			c.Whisper.RemoteURL = "://bad-url"
		}, true},
		{"remote mode rejects non-http scheme", func(c *Config) {
			c.Whisper.RemoteURL = "ftp://example.com"
		}, true},
		{"server port zero", func(c *Config) { c.Server.Port = 0 }, true},
		{"server port too big", func(c *Config) { c.Server.Port = 65536 }, true},
		{"whisper port zero", func(c *Config) { c.Whisper.Port = 0 }, true},
		{"whisper port too big", func(c *Config) { c.Whisper.Port = 65536 }, true},
		{"empty server host", func(c *Config) { c.Server.Host = "" }, true},
		{"empty whisper host", func(c *Config) { c.Whisper.Host = "" }, true},
		{"port boundaries ok", func(c *Config) { c.Server.Port = 1; c.Whisper.Port = 65535 }, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := base(t)
			tt.mutate(c)
			err := c.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestListenAddr(t *testing.T) {
	c := &Config{Server: ServerConfig{Host: "0.0.0.0", Port: 8000}}
	if got := c.ListenAddr(); got != "0.0.0.0:8000" {
		t.Errorf("ListenAddr() = %q, want 0.0.0.0:8000", got)
	}
}

func TestWhisperURL(t *testing.T) {
	c := &Config{Whisper: WhisperConfig{Host: "127.0.0.1", Port: 8080}}
	if got := c.WhisperURL(); got != "http://127.0.0.1:8080" {
		t.Errorf("WhisperURL() = %q, want http://127.0.0.1:8080", got)
	}
}
