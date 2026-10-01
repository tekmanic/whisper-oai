// Package config loads, merges, and validates whisper-oai configuration.
//
// Precedence (highest wins): explicit flags recorded via SetFlag, environment
// variables (WHISPER_OAI_*), the config file, then built-in defaults.
package config

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/spf13/viper"
)

// Config is the root whisper-oai configuration.
type Config struct {
	Server  ServerConfig  `mapstructure:"server"`
	Whisper WhisperConfig `mapstructure:"whisper"`
	Audio   AudioConfig   `mapstructure:"audio"`
	Log     LogConfig     `mapstructure:"log"`
}

// ServerConfig configures the public OpenAI-compatible HTTP server.
type ServerConfig struct {
	Host      string `mapstructure:"host"`       // default "0.0.0.0"
	Port      int    `mapstructure:"port"`       // default 8000
	APIKey    string `mapstructure:"api_key"`    // default "" (no auth)
	ModelName string `mapstructure:"model_name"` // default "whisper-1"
}

// WhisperConfig configures the whisper-server backend process.
type WhisperConfig struct {
	Bin       string   `mapstructure:"bin"`        // default /opt/whisper.cpp/build/bin/whisper-server
	Model     string   `mapstructure:"model"`      // default /opt/whisper.cpp/models/ggml-large-v3-turbo.bin
	RemoteURL string   `mapstructure:"remote_url"` // default "" (if set, use remote backend and do not spawn local process)
	Device    int      `mapstructure:"device"`     // default 0
	FlashAttn bool     `mapstructure:"flash_attn"` // default true
	Host      string   `mapstructure:"host"`       // default "127.0.0.1" (loopback only)
	Port      int      `mapstructure:"port"`       // default 8080
	Threads   int      `mapstructure:"threads"`    // default 0 (auto)
	Language  string   `mapstructure:"language"`   // default "auto"
	ExtraArgs []string `mapstructure:"extra_args"` // default nil
}

// AudioConfig configures audio handling for uploads.
type AudioConfig struct {
	FfmpegBin   string `mapstructure:"ffmpeg_bin"`    // default "ffmpeg"
	TempDir     string `mapstructure:"temp_dir"`      // default "" (os.TempDir)
	MaxUploadMB int    `mapstructure:"max_upload_mb"` // default 2048
}

// LogConfig configures logging.
type LogConfig struct {
	Level string `mapstructure:"level"` // default "info"
}

var (
	flagMu        sync.Mutex
	flagOverrides = map[string]string{}
)

// SetFlag records a CLI flag override (key = dotted, e.g. "server.port",
// value string). Call before Load.
func SetFlag(key, value string) {
	flagMu.Lock()
	defer flagMu.Unlock()
	flagOverrides[key] = value
}

// Load reads configuration from (in precedence order): explicit flags set via
// SetFlag, environment variables (WHISPER_OAI_*), the file at path (if
// non-empty and exists), then built-in defaults. It returns a fully-populated
// *Config.
func Load(path string) (*Config, error) {
	v := viper.New()
	v.SetEnvPrefix("WHISPER_OAI")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	setDefaults(v)

	if path != "" {
		if _, err := os.Stat(path); err == nil {
			v.SetConfigFile(path)
			if err := v.ReadInConfig(); err != nil {
				return nil, fmt.Errorf("read config file %s: %w", path, err)
			}
		}
	}

	flagMu.Lock()
	for key, value := range flagOverrides {
		v.Set(key, value)
	}
	flagMu.Unlock()

	var c Config
	if err := v.Unmarshal(&c); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}
	return &c, nil
}

// setDefaults registers the built-in defaults (matching
// config/whisper-oai.example.yaml). whisper.extra_args intentionally has no
// default so it stays nil unless explicitly set.
func setDefaults(v *viper.Viper) {
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8000)
	v.SetDefault("server.api_key", "")
	v.SetDefault("server.model_name", "whisper-1")
	v.SetDefault("whisper.bin", "/opt/whisper.cpp/build/bin/whisper-server")
	v.SetDefault("whisper.model", "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin")
	v.SetDefault("whisper.remote_url", "")
	v.SetDefault("whisper.device", 0)
	v.SetDefault("whisper.flash_attn", true)
	v.SetDefault("whisper.host", "127.0.0.1")
	v.SetDefault("whisper.port", 8080)
	v.SetDefault("whisper.threads", 0)
	v.SetDefault("whisper.language", "auto")
	v.SetDefault("audio.ffmpeg_bin", "ffmpeg")
	v.SetDefault("audio.temp_dir", "")
	v.SetDefault("audio.max_upload_mb", 2048)
	v.SetDefault("log.level", "info")
}

// Validate checks required fields and value ranges for both backend modes:
// local process mode (whisper.remote_url empty) and remote backend mode
// (whisper.remote_url set). It does NOT stat local bin/model files (so tests
// pass without a real install); the serve command logs a warning if they are
// missing.
func (c *Config) Validate() error {
	var problems []string
	if c.Server.Host == "" {
		problems = append(problems, "server.host must not be empty")
	}
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		problems = append(problems, fmt.Sprintf("server.port must be in [1,65535], got %d", c.Server.Port))
	}
	if remoteURL := strings.TrimSpace(c.Whisper.RemoteURL); remoteURL != "" {
		u, err := url.Parse(remoteURL)
		if err != nil {
			problems = append(problems, fmt.Sprintf("whisper.remote_url must be a valid URL, got %q", c.Whisper.RemoteURL))
		} else {
			if u.Scheme != "http" && u.Scheme != "https" {
				problems = append(problems, fmt.Sprintf("whisper.remote_url must use http or https, got %q", u.Scheme))
			}
			if u.Host == "" {
				problems = append(problems, "whisper.remote_url must include a host")
			}
		}
	} else {
		if c.Whisper.Bin == "" {
			problems = append(problems, "whisper.bin must not be empty")
		}
		if c.Whisper.Model == "" {
			problems = append(problems, "whisper.model must not be empty")
		}
		if c.Whisper.Host == "" {
			problems = append(problems, "whisper.host must not be empty")
		}
		if c.Whisper.Port < 1 || c.Whisper.Port > 65535 {
			problems = append(problems, fmt.Sprintf("whisper.port must be in [1,65535], got %d", c.Whisper.Port))
		}
	}
	if len(problems) > 0 {
		return fmt.Errorf("invalid config: %s", strings.Join(problems, "; "))
	}
	return nil
}

// ListenAddr returns "host:port" for the public OpenAI-compatible server.
func (c *Config) ListenAddr() string {
	return net.JoinHostPort(c.Server.Host, strconv.Itoa(c.Server.Port))
}

// WhisperURL returns "http://host:port" for the whisper-server backend.
func (c *Config) WhisperURL() string {
	if u := strings.TrimSpace(c.Whisper.RemoteURL); u != "" {
		return strings.TrimRight(u, "/")
	}
	return "http://" + net.JoinHostPort(c.Whisper.Host, strconv.Itoa(c.Whisper.Port))
}
