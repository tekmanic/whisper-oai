package download

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// writeFakeYTDLP writes a POSIX-sh fake yt-dlp into dir and returns its path.
// The fake records its argv to the file named by $FAKE_YTDLP_ARGV (if set),
// resolves the -o template by replacing the literal "%(ext)s" with ext,
// writes "fake media bytes" to the result, and prints the final path on
// stdout (mimicking --print after_move:filepath).
//
// The %(ext)s replacement uses a sed pipeline: the ${var//pat/rep} form from
// the spec is a bashism that dash (this environment's /bin/sh) rejects.
func writeFakeYTDLP(t *testing.T, dir, name, ext string) string {
	t.Helper()
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
out=$(printf '%s' "$out" | sed 's/%(ext)s/__FAKE_EXT__/')
printf 'fake media bytes' > "$out"
echo "$out"
`
	body = strings.ReplaceAll(body, "__FAKE_EXT__", ext)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake yt-dlp: %v", err)
	}
	return path
}

func TestDownloadSuccess(t *testing.T) {
	dir := t.TempDir()
	fake := writeFakeYTDLP(t, dir, "fake-yt-dlp", "mp4")
	argvFile := filepath.Join(dir, "argv.txt")
	t.Setenv("FAKE_YTDLP_ARGV", argvFile)

	d := New(Options{
		Bin:     fake,
		Timeout: time.Minute,
		Format:  "bv*+ba/b",
		TempDir: t.TempDir(),
	}, nil)

	const rawURL = "https://www.youtube.com/watch?v=abc123"
	path, filename, err := d.Download(context.Background(), rawURL)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("downloaded file %q does not exist: %v", path, err)
	}
	if !strings.HasSuffix(filename, ".mp4") {
		t.Errorf("filename = %q, want suffix .mp4", filename)
	}
	if filepath.Base(path) != filename {
		t.Errorf("filename = %q, want %q (base of path)", filename, filepath.Base(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %q: %v", path, err)
	}
	if string(data) != "fake media bytes" {
		t.Errorf("content = %q, want %q", data, "fake media bytes")
	}

	argvData, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("read recorded argv: %v", err)
	}
	argv := strings.Split(strings.TrimSpace(string(argvData)), "\n")
	for _, want := range []string{"--no-playlist", "-f", "bv*+ba/b", "--print", "after_move:filepath"} {
		if !slices.Contains(argv, want) {
			t.Errorf("argv missing %q: %v", want, argv)
		}
	}
	if got := argv[len(argv)-1]; got != rawURL {
		t.Errorf("last argv = %q, want %q", got, rawURL)
	}
}

func TestDownloadInvalidURL(t *testing.T) {
	d := New(Options{}, nil)
	for _, rawURL := range []string{"", "notaurl", "file:///etc/passwd", "ftp://example.com/a.mp4"} {
		_, _, err := d.Download(context.Background(), rawURL)
		if err == nil {
			t.Errorf("Download(%q): expected error, got nil", rawURL)
			continue
		}
		if !errors.Is(err, ErrInvalidURL) {
			t.Errorf("Download(%q): error = %v, want ErrInvalidURL", rawURL, err)
		}
	}
}

func TestDownloadHostAllowlist(t *testing.T) {
	dir := t.TempDir()
	fake := writeFakeYTDLP(t, dir, "fake-yt-dlp", "mp4")

	tests := []struct {
		name    string
		url     string
		wantErr error
	}{
		{"exact host", "https://example.com/v", nil},
		{"wildcard subdomain", "https://www.youtube.com/watch", nil},
		{"wildcard apex", "https://youtube.com/watch", nil},
		{"disallowed host", "https://evil.com/v", ErrHostNotAllowed},
		{"suffix lookalike", "https://notexample.com/v", ErrHostNotAllowed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := New(Options{
				Bin:          fake,
				Timeout:      time.Minute,
				AllowedHosts: []string{"example.com", "*.youtube.com"},
				TempDir:      t.TempDir(),
			}, nil)
			_, _, err := d.Download(context.Background(), tt.url)
			if tt.wantErr == nil {
				if err != nil {
					t.Fatalf("Download(%q): %v", tt.url, err)
				}
				return
			}
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Download(%q): error = %v, want %v", tt.url, err, tt.wantErr)
			}
		})
	}
}

func TestDownloadFailure(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "fake-yt-dlp")
	body := "#!/bin/sh\necho 'ERROR: [youtube] x: Unable to download webpage: HTTP 403' >&2\nexit 1\n"
	if err := os.WriteFile(fake, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake yt-dlp: %v", err)
	}

	d := New(Options{Bin: fake, Timeout: time.Minute, TempDir: t.TempDir()}, nil)
	_, _, err := d.Download(context.Background(), "https://example.com/v")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if errors.Is(err, ErrInvalidURL) || errors.Is(err, ErrHostNotAllowed) {
		t.Errorf("error = %v, must not be a validation error", err)
	}
	if !strings.Contains(err.Error(), "Unable to download webpage") {
		t.Errorf("error = %q, want it to contain %q", err, "Unable to download webpage")
	}
}

func TestDownloadTimeout(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "fake-yt-dlp")
	body := "#!/bin/sh\nsleep 2\nexit 0\n"
	if err := os.WriteFile(fake, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake yt-dlp: %v", err)
	}

	d := New(Options{Bin: fake, Timeout: 200 * time.Millisecond, TempDir: t.TempDir()}, nil)
	_, _, err := d.Download(context.Background(), "https://example.com/v")
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !strings.Contains(err.Error(), "timed out") && !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %q, want timeout indication", err)
	}
}

func TestDownloadMissingBinary(t *testing.T) {
	d := New(Options{Bin: "/nonexistent/yt-dlp-does-not-exist", Timeout: time.Minute, TempDir: t.TempDir()}, nil)
	_, _, err := d.Download(context.Background(), "https://example.com/v")
	if err == nil {
		t.Fatal("expected error for missing binary, got nil")
	}
}
