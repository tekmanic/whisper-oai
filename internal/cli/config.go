package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/tekmanic/whisper-oai/internal/config"
)

// exampleConfig is the content written by `config init`. It mirrors
// config/whisper-oai.example.yaml.
const exampleConfig = `# whisper-oai configuration
#
# Copy this file to config/whisper-oai.yaml (or point --config at your own)
# and edit. Every value can also be set via a flag on ` + "`" + `whisper-oai serve` + "`" + `
# or an environment variable (WHISPER_OAI_<SECTION>_<KEY>, e.g.
# WHISPER_OAI_SERVER_PORT). Precedence: flag > env > config file > default.

# ---------------------------------------------------------------------------
# External OpenAI-compatible API server (what clients talk to)
# ---------------------------------------------------------------------------
server:
  # Host/interface the proxy listens on.
  host: "0.0.0.0"
  # Port the proxy listens on. This is the "user's specified port".
  port: 8000
  # Optional. If set, clients must send "Authorization: Bearer <api_key>".
  # Empty means no authentication (fine for localhost-only use).
  api_key: ""
  # Model name advertised by /v1/models and echoed in responses.
  model_name: "whisper-1"

# ---------------------------------------------------------------------------
# whisper.cpp whisper-server (the backend the proxy drives)
# ---------------------------------------------------------------------------
whisper:
  # Path to the whisper-server binary.
  bin: "/opt/whisper.cpp/build/bin/whisper-server"
  # Path to the GGML model file.
  model: "/opt/whisper.cpp/models/ggml-large-v3-turbo.bin"
	# Optional remote whisper-server base URL. If set, whisper-oai will use this
	# backend directly and will NOT start a local whisper-server process.
	# Example: "http://127.0.0.1:8080"
	remote_url: ""
  # GPU device index (whisper-server -dev / --device).
  device: 0
  # Enable flash attention (whisper-server --flash-attn).
  flash_attn: true
  # Host whisper-server binds to. Keep 127.0.0.1 so ONLY the Go proxy can
  # reach it (loopback). Do not expose this port.
  host: "127.0.0.1"
  # Port whisper-server binds to (internal). The proxy talks to this.
  # Kept separate from server.port so the two never collide.
  port: 8080
  # Number of CPU threads for whisper-server (-t). 0 = auto (min(4, cores)).
  threads: 0
  # Default language for inference ("en", "auto", ...). Clients can override
  # per-request with the ` + "`" + `language` + "`" + ` field.
  language: "auto"
  # Extra arguments appended verbatim to the whisper-server command line.
  # Use for anything not covered above, e.g. ["-t", "8", "--no-gpu"].
  extra_args: []

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log:
  # debug | info | warn | error
  level: "info"
`

// newConfigCmd returns the config parent command with its subcommands.
func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage the whisper-oai configuration file",
	}
	cmd.AddCommand(newConfigInitCmd())
	cmd.AddCommand(newConfigGetCmd())
	cmd.AddCommand(newConfigSetCmd())
	cmd.AddCommand(newConfigPathCmd())
	return cmd
}

// newConfigInitCmd returns the `config init` subcommand.
func newConfigInitCmd() *cobra.Command {
	var force bool
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Write the example configuration file",
		Long: `Writes the example configuration to the --config path (default
config/whisper-oai.yaml), creating parent directories as needed. Refuses to
overwrite an existing file unless --force is given.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfgPath, err := cmd.Flags().GetString("config")
			if err != nil {
				return fmt.Errorf("read --config flag: %w", err)
			}
			if _, err := os.Stat(cfgPath); err == nil {
				if !force {
					return fmt.Errorf("config file already exists at %s (use --force to overwrite)", cfgPath)
				}
			} else if !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("check %s: %w", cfgPath, err)
			}
			if err := os.MkdirAll(filepath.Dir(cfgPath), 0o755); err != nil {
				return fmt.Errorf("create %s: %w", filepath.Dir(cfgPath), err)
			}
			if err := os.WriteFile(cfgPath, []byte(exampleConfig), 0o644); err != nil {
				return fmt.Errorf("write %s: %w", cfgPath, err)
			}
			fmt.Fprintln(cmd.OutOrStdout(), "wrote", cfgPath)
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the config file if it already exists")
	return cmd
}

// newConfigGetCmd returns the `config get` subcommand.
func newConfigGetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get <key>",
		Short: "Print the value of a config key (dotted, e.g. server.port)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfgPath, err := cmd.Flags().GetString("config")
			if err != nil {
				return fmt.Errorf("read --config flag: %w", err)
			}
			cfg, err := config.Load(cfgPath)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}
			value, ok := lookupValue(cfg, args[0])
			if !ok {
				return fmt.Errorf("config key %q not found", args[0])
			}
			fmt.Fprintln(cmd.OutOrStdout(), value)
			return nil
		},
	}
}

// newConfigSetCmd returns the `config set` subcommand.
func newConfigSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set <key> <value>",
		Short: "Set a config key (dotted, e.g. server.port) in the config file",
		Long: `Sets a value in the config file, creating the file from built-in
defaults if it does not exist yet. The value is parsed for the key's type:
integer for ports/device/threads, boolean for whisper.flash_attn, and a
comma-separated list for whisper.extra_args.`,
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfgPath, err := cmd.Flags().GetString("config")
			if err != nil {
				return fmt.Errorf("read --config flag: %w", err)
			}
			cfg, err := config.Load(cfgPath)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}
			if err := setValue(cfg, args[0], args[1]); err != nil {
				return err
			}
			if err := writeConfigFile(cfgPath, cfg); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "set %s = %s in %s\n", args[0], args[1], cfgPath)
			return nil
		},
	}
}

// newConfigPathCmd returns the `config path` subcommand.
func newConfigPathCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "path",
		Short: "Print the resolved config file path",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfgPath, err := cmd.Flags().GetString("config")
			if err != nil {
				return fmt.Errorf("read --config flag: %w", err)
			}
			fmt.Fprintln(cmd.OutOrStdout(), cfgPath)
			return nil
		},
	}
}

// lookupValue returns the string form of the value for a dotted config key,
// or ok=false if the key is unknown.
func lookupValue(cfg *config.Config, key string) (string, bool) {
	switch key {
	case "server.host":
		return cfg.Server.Host, true
	case "server.port":
		return strconv.Itoa(cfg.Server.Port), true
	case "server.api_key":
		return cfg.Server.APIKey, true
	case "server.model_name":
		return cfg.Server.ModelName, true
	case "whisper.bin":
		return cfg.Whisper.Bin, true
	case "whisper.model":
		return cfg.Whisper.Model, true
	case "whisper.remote_url":
		return cfg.Whisper.RemoteURL, true
	case "whisper.device":
		return strconv.Itoa(cfg.Whisper.Device), true
	case "whisper.flash_attn":
		return strconv.FormatBool(cfg.Whisper.FlashAttn), true
	case "whisper.host":
		return cfg.Whisper.Host, true
	case "whisper.port":
		return strconv.Itoa(cfg.Whisper.Port), true
	case "whisper.threads":
		return strconv.Itoa(cfg.Whisper.Threads), true
	case "whisper.language":
		return cfg.Whisper.Language, true
	case "whisper.extra_args":
		return strings.Join(cfg.Whisper.ExtraArgs, " "), true
	case "log.level":
		return cfg.Log.Level, true
	}
	return "", false
}

// setValue assigns a parsed value to a dotted config key, returning an error
// for unknown keys or values that do not match the key's type.
func setValue(cfg *config.Config, key, value string) error {
	switch key {
	case "server.host":
		cfg.Server.Host = value
	case "server.port":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("config key %q expects an integer, got %q", key, value)
		}
		cfg.Server.Port = n
	case "server.api_key":
		cfg.Server.APIKey = value
	case "server.model_name":
		cfg.Server.ModelName = value
	case "whisper.bin":
		cfg.Whisper.Bin = value
	case "whisper.model":
		cfg.Whisper.Model = value
	case "whisper.remote_url":
		cfg.Whisper.RemoteURL = value
	case "whisper.device":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("config key %q expects an integer, got %q", key, value)
		}
		cfg.Whisper.Device = n
	case "whisper.flash_attn":
		b, err := strconv.ParseBool(value)
		if err != nil {
			return fmt.Errorf("config key %q expects a boolean, got %q", key, value)
		}
		cfg.Whisper.FlashAttn = b
	case "whisper.host":
		cfg.Whisper.Host = value
	case "whisper.port":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("config key %q expects an integer, got %q", key, value)
		}
		cfg.Whisper.Port = n
	case "whisper.threads":
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("config key %q expects an integer, got %q", key, value)
		}
		cfg.Whisper.Threads = n
	case "whisper.language":
		cfg.Whisper.Language = value
	case "whisper.extra_args":
		var args []string
		for _, part := range strings.Split(value, ",") {
			part = strings.TrimSpace(part)
			if part != "" {
				args = append(args, part)
			}
		}
		cfg.Whisper.ExtraArgs = args
	case "log.level":
		cfg.Log.Level = value
	default:
		return fmt.Errorf("config key %q not found", key)
	}
	return nil
}

// writeConfigFile writes the full config back to path as YAML. The config
// struct only carries mapstructure tags, so the YAML is produced through a
// fresh viper instance seeded with every known key.
func writeConfigFile(path string, cfg *config.Config) error {
	v := viper.New()
	v.SetConfigFile(path)
	v.SetConfigType("yaml")
	extra := cfg.Whisper.ExtraArgs
	if extra == nil {
		extra = []string{}
	}
	v.Set("server.host", cfg.Server.Host)
	v.Set("server.port", cfg.Server.Port)
	v.Set("server.api_key", cfg.Server.APIKey)
	v.Set("server.model_name", cfg.Server.ModelName)
	v.Set("whisper.bin", cfg.Whisper.Bin)
	v.Set("whisper.model", cfg.Whisper.Model)
	v.Set("whisper.remote_url", cfg.Whisper.RemoteURL)
	v.Set("whisper.device", cfg.Whisper.Device)
	v.Set("whisper.flash_attn", cfg.Whisper.FlashAttn)
	v.Set("whisper.host", cfg.Whisper.Host)
	v.Set("whisper.port", cfg.Whisper.Port)
	v.Set("whisper.threads", cfg.Whisper.Threads)
	v.Set("whisper.language", cfg.Whisper.Language)
	v.Set("whisper.extra_args", extra)
	v.Set("log.level", cfg.Log.Level)
	if err := v.WriteConfigAs(path); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}
