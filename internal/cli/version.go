package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

// version is the build version reported by the version command. It is set by
// the main package (ldflags -X main.version=...) via SetVersion; defaults to
// "dev".
var version = "dev"

// SetVersion overrides the version reported by the version command.
func SetVersion(v string) {
	if v != "" {
		version = v
	}
}

// newVersionCmd returns the version subcommand.
func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the whisper-oai version",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Fprintln(cmd.OutOrStdout(), "whisper-oai version", version)
		},
	}
}
