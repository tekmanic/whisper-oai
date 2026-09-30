// Package cli implements the whisper-oai command-line interface: a root
// command with serve, config and version subcommands.
package cli

import (
	"github.com/spf13/cobra"
)

// defaultConfigPath is the default location of the YAML config file.
const defaultConfigPath = "config/whisper-oai.yaml"

// NewRootCmd returns the root cobra command with all subcommands attached.
func NewRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "whisper-oai",
		Short: "OpenAI-compatible API server for whisper.cpp",
		Long: `whisper-oai wraps whisper.cpp's whisper-server and exposes it as an
OpenAI-compatible HTTP API.

It starts and supervises a whisper-server child process, then serves
/v1/audio/transcriptions, /v1/audio/translations and /v1/models on a public
port so any OpenAI client can transcribe audio with whisper.cpp.`,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	cmd.PersistentFlags().String("config", defaultConfigPath, "path to the YAML config file")
	cmd.AddCommand(newServeCmd())
	cmd.AddCommand(newConfigCmd())
	cmd.AddCommand(newVersionCmd())
	return cmd
}

// Execute builds the root command and runs it; returns the error (main maps
// this to os.Exit(1)).
func Execute() error {
	return NewRootCmd().Execute()
}
