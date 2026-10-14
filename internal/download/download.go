// Package download fetches remote media (video/audio) from a URL using
// yt-dlp, so the transcription API can accept a link instead of a file
// upload.
package download

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ErrInvalidURL is returned when the URL is malformed or not http/https.
var ErrInvalidURL = errors.New("invalid URL")

// ErrHostNotAllowed is returned when the URL's host is not in the configured
// allowlist.
var ErrHostNotAllowed = errors.New("host not allowed")

// Options configures a Downloader.
type Options struct {
	Bin          string        // yt-dlp binary ("" → "yt-dlp")
	Timeout      time.Duration // per-download timeout (<= 0 → 10m)
	Format       string        // yt-dlp -f format selector ("" → "bv*+ba/b")
	ExtraArgs    []string      // extra args appended after the built-in args, before the URL
	AllowedHosts []string      // host allowlist; empty/nil = allow all. Entries are exact hosts ("youtube.com") or wildcard suffixes ("*.youtube.com").
	TempDir      string        // directory for downloaded files ("" → os.TempDir)
}

// Downloader downloads media from URLs via yt-dlp.
type Downloader struct {
	opts Options
	log  *slog.Logger
}

// New builds a Downloader, normalizing empty options to their defaults. A nil
// log is tolerated (treated as a no-op logger).
func New(opts Options, log *slog.Logger) *Downloader {
	if opts.Bin == "" {
		opts.Bin = "yt-dlp"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 10 * time.Minute
	}
	if opts.Format == "" {
		opts.Format = "bv*+ba/b"
	}
	return &Downloader{opts: opts, log: log}
}

// maxErrOutput bounds how many bytes of yt-dlp output are embedded in
// returned errors.
const maxErrOutput = 2048

// Download fetches the media at rawURL into a temp file and returns the file
// path and its base filename. The caller MUST os.Remove the returned path.
//
// Validation failures wrap ErrInvalidURL or ErrHostNotAllowed. Download
// failures wrap the exec error and include the tail of yt-dlp stderr.
func (d *Downloader) Download(ctx context.Context, rawURL string) (string, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", "", fmt.Errorf("%w: %q must use http or https", ErrInvalidURL, rawURL)
	}
	host := strings.ToLower(u.Host)
	if len(d.opts.AllowedHosts) > 0 && !hostAllowed(host, d.opts.AllowedHosts) {
		return "", "", fmt.Errorf("%w: %q (allowed: %s)", ErrHostNotAllowed, host, strings.Join(d.opts.AllowedHosts, ", "))
	}

	var name [8]byte
	if _, err := rand.Read(name[:]); err != nil {
		return "", "", fmt.Errorf("download %q: %w", rawURL, err)
	}
	base := "whisper-oai-dl-" + hex.EncodeToString(name[:])
	tempDir := d.opts.TempDir
	if tempDir == "" {
		tempDir = os.TempDir()
	}
	outTemplate := filepath.Join(tempDir, base+".%(ext)s")

	args := []string{
		"--no-playlist", "--no-warnings",
		"-f", d.opts.Format,
		"-o", outTemplate,
		"--print", "after_move:filepath",
	}
	args = append(args, d.opts.ExtraArgs...)
	args = append(args, rawURL)

	if d.log != nil {
		d.log.Debug("downloading media from URL", "url", rawURL, "bin", d.opts.Bin)
	}

	ctx2, cancel := context.WithTimeout(ctx, d.opts.Timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx2, d.opts.Bin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx2.Err() == context.DeadlineExceeded {
			return "", "", fmt.Errorf("download %q: timed out after %s", rawURL, d.opts.Timeout)
		}
		if tail := strings.TrimSpace(stderr.String()); tail != "" {
			return "", "", fmt.Errorf("download %q: %w: %s", rawURL, err, truncate(tail, maxErrOutput))
		}
		return "", "", fmt.Errorf("download %q: %w", rawURL, err)
	}

	// yt-dlp's --print after_move:filepath output: the final path is the last
	// non-empty line of stdout.
	path := lastNonEmptyLine(stdout.String())
	if path != "" {
		if _, err := os.Stat(path); err != nil {
			path = "" // parsed path unusable; fall back to glob
		}
	}
	if path == "" {
		matches, _ := filepath.Glob(filepath.Join(tempDir, base+".*"))
		for _, m := range matches {
			if st, err := os.Stat(m); err == nil && st.Mode().IsRegular() {
				path = m
				break
			}
		}
	}
	if path == "" {
		if tail := strings.TrimSpace(stderr.String()); tail != "" {
			return "", "", fmt.Errorf("download %q: no output file found: %s", rawURL, truncate(tail, maxErrOutput))
		}
		return "", "", fmt.Errorf("download %q: no output file found", rawURL)
	}

	var size int64
	if st, err := os.Stat(path); err == nil {
		size = st.Size()
	}
	if d.log != nil {
		d.log.Debug("media download complete", "url", rawURL, "path", path, "size", size)
	}
	return path, filepath.Base(path), nil
}

// hostAllowed reports whether host (already lowercased) matches any entry in
// allowed. An entry of the form "*.example.com" matches the apex domain
// ("example.com") and any subdomain ("www.example.com"), but not a different
// domain that merely ends with the same string ("notexample.com"); any other
// entry must equal the host exactly.
func hostAllowed(host string, allowed []string) bool {
	for _, entry := range allowed {
		e := strings.ToLower(entry)
		if strings.HasPrefix(e, "*.") {
			suffix := e[2:] // strip the leading "*."
			if host == suffix || strings.HasSuffix(host, "."+suffix) {
				return true
			}
		} else if host == e {
			return true
		}
	}
	return false
}

// lastNonEmptyLine returns the last non-empty line of s, or "" if there is
// none.
func lastNonEmptyLine(s string) string {
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}

// truncate shortens s to at most n bytes, keeping the tail (where yt-dlp
// reports the actual error) and appending an ellipsis when cut.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}
