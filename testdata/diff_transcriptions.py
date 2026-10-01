#!/usr/bin/env python3
"""Run two transcription curl requests and diff their outputs."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path


def run_curl(url: str, media_path: Path, model: str) -> str:
    cmd = [
        "curl",
        "-sS",
        f"{url}/v1/audio/transcriptions",
        "-F",
        f"file=@{media_path}",
        "-F",
        f"model={model}",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl failed for {media_path.name} (exit={proc.returncode}):\n{proc.stderr.strip()}"
        )

    return proc.stdout.strip()


def pretty_json_or_raw(text: str) -> list[str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines() or [text]
    return json.dumps(obj, indent=2, sort_keys=True).splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare whisper transcription outputs for audio.wav and video.mp4"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for whisper-oai server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="whisper-1",
        help="Model name to pass to the API (default: whisper-1)",
    )
    args = parser.parse_args()

    testdata_dir = Path(__file__).resolve().parent
    audio_path = testdata_dir / "audio.wav"
    video_path = testdata_dir / "video.mp4"

    if not audio_path.exists() or not video_path.exists():
        print("Expected audio.wav and video.mp4 in testdata/", file=sys.stderr)
        return 2

    try:
        audio_out = run_curl(args.url, audio_path, args.model)
        video_out = run_curl(args.url, video_path, args.model)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    audio_lines = pretty_json_or_raw(audio_out)
    video_lines = pretty_json_or_raw(video_out)

    if audio_lines == video_lines:
        print("Outputs are identical.")
        # print("audio.wav output:")
        # print("\n".join(audio_lines))
        return 0

    print("Outputs differ. Unified diff:")
    diff = difflib.unified_diff(
        audio_lines,
        video_lines,
        fromfile="audio.wav",
        tofile="video.mp4",
        lineterm="",
    )
    print("\n".join(diff))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
