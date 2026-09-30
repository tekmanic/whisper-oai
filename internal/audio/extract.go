// Package audio prepares uploaded media for whisper-server.
//
// whisper-server decodes raw audio (wav, mp3, ogg, flac, opus) but not video
// containers, so this package detects video uploads, spools them to disk, and
// runs ffmpeg to extract a 16 kHz mono PCM WAV that whisper.cpp consumes
// natively.
package audio

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// videoExtensions is the set of filename extensions (lowercase, without the
// leading dot) that identify a video container needing audio extraction.
var videoExtensions = map[string]bool{
	"mp4":  true,
	"m4v":  true,
	"mov":  true,
	"mkv":  true,
	"webm": true,
	"avi":  true,
	"wmv":  true,
	"flv":  true,
	"mpg":  true,
	"mpeg": true,
	"3gp":  true,
	"ts":   true,
	"m2ts": true,
	"ogv":  true,
}

// IsVideo reports whether the uploaded file is a video container that needs
// audio extraction, based on filename extension or Content-Type.
// Video extensions: mp4, m4v, mov, mkv, webm, avi, wmv, flv, mpg, mpeg,
// 3gp, ts, m2ts, ogv. Content-Type "video/*" also matches.
func IsVideo(filename, contentType string) bool {
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(filename)), ".")
	if videoExtensions[ext] {
		return true
	}
	return strings.HasPrefix(strings.ToLower(contentType), "video/")
}

// Spool copies the uploaded bytes to a temp file so large uploads stream to
// disk (not memory) and ffmpeg can address them by path. The temp file is
// created in tempDir ("" → os.TempDir) with a unique name. Returns the path;
// the caller MUST os.Remove it.
func Spool(src io.Reader, filename, tempDir string) (string, error) {
	f, err := os.CreateTemp(tempDir, "whisper-oai-upload-*")
	if err != nil {
		return "", fmt.Errorf("spool %q: %w", filename, err)
	}
	path := f.Name()
	if _, err := io.Copy(f, src); err != nil {
		f.Close()
		os.Remove(path)
		return "", fmt.Errorf("spool %q: %w", filename, err)
	}
	if err := f.Close(); err != nil {
		os.Remove(path)
		return "", fmt.Errorf("spool %q: %w", filename, err)
	}
	return path, nil
}

// maxFFmpegErrOutput bounds how many bytes of ffmpeg output are embedded in
// returned errors.
const maxFFmpegErrOutput = 2048

// Extract runs ffmpeg to pull the audio track out of a video file and write
// a 16 kHz mono PCM WAV (the format whisper.cpp consumes natively).
// Command: <ffmpegBin> -nostdin -hide_banner -loglevel error -y -i <srcPath>
//
//	-vn -ac 1 -ar 16000 -c:a pcm_s16le <dst.wav>
//
// The output .wav is created in tempDir ("" → os.TempDir) with a unique name.
// Returns the wav path; the caller MUST os.Remove it. Non-zero exit or a
// missing ffmpeg binary → error (include ffmpeg stderr in the error).
func Extract(ctx context.Context, srcPath, ffmpegBin, tempDir string, log *slog.Logger) (string, error) {
	out, err := os.CreateTemp(tempDir, "whisper-oai-audio-*.wav")
	if err != nil {
		return "", fmt.Errorf("extract %q: %w", srcPath, err)
	}
	out.Close()
	dst := out.Name()

	args := []string{
		"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
		"-i", srcPath,
		"-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
		dst,
	}
	if log != nil {
		log.Debug("extracting audio with ffmpeg", "bin", ffmpegBin, "args", args)
	}

	cmd := exec.CommandContext(ctx, ffmpegBin, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		os.Remove(dst)
		if msg := strings.TrimSpace(string(output)); msg != "" {
			return "", fmt.Errorf("extract %q: %w: %s", srcPath, err, truncate(msg, maxFFmpegErrOutput))
		}
		return "", fmt.Errorf("extract %q: %w", srcPath, err)
	}
	return dst, nil
}

// truncate shortens s to at most n bytes, appending an ellipsis when cut.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
