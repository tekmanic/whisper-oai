package audio

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestIsVideo(t *testing.T) {
	tests := []struct {
		name        string
		filename    string
		contentType string
		want        bool
	}{
		{"mp4", "video.mp4", "", true},
		{"MOV case-insensitive", "movie.MOV", "", true},
		{"mkv", "clip.mkv", "", true},
		{"webm", "stream.webm", "", true},
		{"ts", "video.ts", "", true},
		{"m2ts", "video.m2ts", "", true},
		{"3gp", "a.3gp", "", true},
		{"ogv", "clip.ogv", "", true},
		{"avi", "video.avi", "", true},
		{"wmv", "video.wmv", "", true},
		{"flv", "video.flv", "", true},
		{"mpg", "video.mpg", "", true},
		{"mpeg", "video.mpeg", "", true},
		{"m4v", "video.m4v", "", true},
		{"wav is audio", "song.wav", "", false},
		{"mp3 is audio", "track.mp3", "", false},
		{"ogg is audio", "audio.ogg", "", false},
		{"flac is audio", "voice.flac", "", false},
		{"opus is audio", "talk.opus", "", false},
		{"m4a is audio", "song.m4a", "", false},
		{"aac is audio", "audio.aac", "", false},
		{"content type video only", "", "video/mp4", true},
		{"content type audio only", "", "audio/wav", false},
		{"content type wins over extension", "notes.txt", "video/mp4", true},
		{"no filename no content type", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsVideo(tt.filename, tt.contentType); got != tt.want {
				t.Errorf("IsVideo(%q, %q) = %v, want %v", tt.filename, tt.contentType, got, tt.want)
			}
		})
	}
}

func TestSpool(t *testing.T) {
	data := []byte("hello, whisper")
	path, err := Spool(bytes.NewReader(data), "upload.bin", t.TempDir())
	if err != nil {
		t.Fatalf("Spool: %v", err)
	}
	defer os.Remove(path)

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read spooled file: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("spooled content = %q, want %q", got, data)
	}
}

func TestSpoolDefaultTempDir(t *testing.T) {
	data := []byte("default temp dir")
	path, err := Spool(bytes.NewReader(data), "upload.bin", "")
	if err != nil {
		t.Fatalf("Spool: %v", err)
	}
	defer os.Remove(path)

	if dir := filepath.Dir(path); dir != os.TempDir() {
		t.Errorf("spooled file in %q, want %q", dir, os.TempDir())
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read spooled file: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("spooled content = %q, want %q", got, data)
	}
}

// writeFakeFFmpeg writes an executable shell script that records its argv to
// argvPath (one argument per line) and writes payload to its last argument
// (the output path). It returns the script path.
func writeFakeFFmpeg(t *testing.T, dir, argvPath, payload string) string {
	t.Helper()
	script := filepath.Join(dir, "ffmpeg.sh")
	body := "#!/bin/sh\n" +
		"printf '%s\\n' \"$@\" > " + strconv.Quote(argvPath) + "\n" +
		"last=\"\"\n" +
		"for a in \"$@\"; do last=\"$a\"; done\n" +
		"printf " + strconv.Quote(payload) + " > \"$last\"\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake ffmpeg: %v", err)
	}
	return script
}

func TestExtract(t *testing.T) {
	dir := t.TempDir()
	argvPath := filepath.Join(dir, "argv.txt")
	ffmpeg := writeFakeFFmpeg(t, dir, argvPath, "RIFFfake")

	src := filepath.Join(dir, "input.mp4")
	if err := os.WriteFile(src, []byte("dummy video bytes"), 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	wav, err := Extract(context.Background(), src, ffmpeg, dir, log)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	defer os.Remove(wav)

	if !strings.HasSuffix(wav, ".wav") {
		t.Errorf("wav path %q does not end in .wav", wav)
	}
	data, err := os.ReadFile(wav)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	if len(data) == 0 {
		t.Error("wav file is empty, want non-empty")
	}

	raw, err := os.ReadFile(argvPath)
	if err != nil {
		t.Fatalf("read recorded argv: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(raw)), "\n")
	want := []string{
		"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
		"-i", src,
		"-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
		wav,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ffmpeg argv mismatch:\n got  %v\n want %v", got, want)
	}
}

func TestExtractError(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "ffmpeg.sh")
	body := "#!/bin/sh\necho 'boom: no audio stream found' >&2\nexit 1\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake ffmpeg: %v", err)
	}
	src := filepath.Join(dir, "input.mp4")
	if err := os.WriteFile(src, []byte("dummy"), 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	_, err := Extract(context.Background(), src, script, dir, log)
	if err == nil {
		t.Fatal("Extract: expected error, got nil")
	}
	if !strings.Contains(err.Error(), "boom: no audio stream found") {
		t.Errorf("error %q does not contain ffmpeg stderr", err)
	}
}

func TestExtractMissingBinary(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.mp4")
	if err := os.WriteFile(src, []byte("dummy"), 0o644); err != nil {
		t.Fatalf("write src: %v", err)
	}

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	_, err := Extract(context.Background(), src, filepath.Join(dir, "no-such-ffmpeg"), dir, log)
	if err == nil {
		t.Fatal("Extract: expected error for missing binary, got nil")
	}
}
