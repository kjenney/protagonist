#!/usr/bin/env python3
"""
Extract audio from a video and transcribe it to plain text and SSML.

Backends
--------
whisper (default)
    Local transcription via OpenAI Whisper.  No cloud credentials needed.
    Install: pip install openai-whisper
    ffmpeg must be on PATH.

aws-transcribe
    Cloud transcription via AWS Transcribe.
    Requires a writable S3 bucket (--bucket) and AWS credentials.
    Install: pip install boto3  (already required by detector.py)

Usage examples
--------------
# Local Whisper, write text + SSML next to the video
python transcribe.py video.mp4

# Choose a larger Whisper model for better accuracy
python transcribe.py video.mp4 --model medium

# AWS Transcribe backend
python transcribe.py video.mp4 --backend aws-transcribe --bucket my-s3-bucket

# Write outputs to explicit paths
python transcribe.py video.mp4 --text-output transcript.txt --ssml-output transcript.ssml

# Print to stdout instead of writing files
python transcribe.py video.mp4 --stdout
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract the audio track from *video_path* and write a 16-kHz mono WAV
    to *audio_path*.  Raises RuntimeError if ffmpeg fails."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                   # drop video stream
        "-acodec", "pcm_s16le",  # 16-bit linear PCM
        "-ar", "16000",          # 16 kHz sample rate (Whisper's native rate)
        "-ac", "1",              # mono
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Whisper transcription (local)
# ---------------------------------------------------------------------------

def _transcribe_whisper(audio_path: str, model_name: str) -> list[dict]:
    """Return a list of word dicts: {word, start, end} using Whisper."""
    try:
        import whisper
    except ImportError:
        print(
            "openai-whisper is not installed.  Run:\n"
            "  pip install openai-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading Whisper model '{model_name}' …", file=sys.stderr)
    model = whisper.load_model(model_name)
    print("Transcribing …", file=sys.stderr)
    result = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })
    return words


# ---------------------------------------------------------------------------
# AWS Transcribe backend
# ---------------------------------------------------------------------------

def _transcribe_aws(
    audio_path: str,
    bucket: str,
    profile: str | None,
    region: str,
) -> list[dict]:
    """Upload audio to S3, run AWS Transcribe, return word list."""
    try:
        import boto3
        import urllib.request
    except ImportError:
        print("boto3 is not installed.  Run:\n  pip install boto3", file=sys.stderr)
        sys.exit(1)

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3", region_name=region)
    transcribe = session.client("transcribe", region_name=region)

    s3_key = f"transcribe-audio-{uuid.uuid4()}.wav"
    job_name = f"transcribe-job-{uuid.uuid4()}"

    print(f"Uploading audio to s3://{bucket}/{s3_key} …", file=sys.stderr)
    s3.upload_file(audio_path, bucket, s3_key)

    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"s3://{bucket}/{s3_key}"},
            MediaFormat="wav",
            LanguageCode="en-US",
            Settings={"EnableWordTimeOffsets": True},
        )

        print("Waiting for AWS Transcribe job to complete …", file=sys.stderr)
        while True:
            resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                break
            if status == "FAILED":
                reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
                raise RuntimeError(f"AWS Transcribe job failed: {reason}")
            time.sleep(5)

        transcript_uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        with urllib.request.urlopen(transcript_uri) as f:
            transcript_data = json.loads(f.read())

    finally:
        s3.delete_object(Bucket=bucket, Key=s3_key)

    words = []
    for item in transcript_data.get("results", {}).get("items", []):
        if item["type"] == "pronunciation":
            words.append({
                "word": item["alternatives"][0]["content"],
                "start": round(float(item["start_time"]), 3),
                "end": round(float(item["end_time"]), 3),
            })
    return words


# ---------------------------------------------------------------------------
# Plain-text output
# ---------------------------------------------------------------------------

def words_to_text(words: list[dict]) -> str:
    """Join words into a readable plain-text string."""
    if not words:
        return ""
    tokens = []
    for i, w in enumerate(words):
        word = w["word"]
        # Attach punctuation-only tokens to the preceding word
        if i > 0 and word in {",", ".", "!", "?", ";", ":", "—", "–", "..."}:
            if tokens:
                tokens[-1] += word
                continue
        tokens.append(word)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# SSML output
# ---------------------------------------------------------------------------

# Gaps (seconds) used to decide sentence / paragraph boundaries in SSML
_SENTENCE_BREAK_S = 0.4   # gap ≥ this → end current <s>, start a new one
_PARAGRAPH_BREAK_S = 1.2  # gap ≥ this → end current <p>, start a new one


def words_to_ssml(words: list[dict]) -> str:
    """Convert a timed word list to an SSML document.

    Structure:
      <speak>
        <p>
          <s>sentence one</s>
          <s>sentence two</s>
        </p>
        <p>
          <s>new paragraph after a long pause</s>
        </p>
      </speak>

    Sentence-ending punctuation (. ! ?) also triggers a new <s>.
    An explicit <break> element is inserted whenever the measured gap
    between two consecutive words falls between the two thresholds.
    """
    speak = ET.Element("speak")

    if not words:
        return _pretty_xml(speak)

    current_para = ET.SubElement(speak, "p")
    current_sent = ET.SubElement(current_para, "s")
    sent_parts: list[str] = []   # text accumulated for the current <s>

    def _flush_sentence() -> None:
        text = " ".join(sent_parts).strip()
        if text:
            current_sent.text = text
        sent_parts.clear()

    sentence_end_punct = {".", "!", "?"}

    for i, w in enumerate(words):
        word = w["word"]

        if i > 0:
            gap = w["start"] - words[i - 1]["end"]

            if gap >= _PARAGRAPH_BREAK_S:
                # Long pause → new paragraph
                _flush_sentence()
                current_para = ET.SubElement(speak, "p")
                current_sent = ET.SubElement(current_para, "s")

            elif gap >= _SENTENCE_BREAK_S:
                # Medium pause → new sentence within same paragraph
                _flush_sentence()
                current_sent = ET.SubElement(current_para, "s")

            else:
                # Short pause → stay in same sentence, add a <break> if notable
                # (gaps under 150 ms are normal speech rhythm, skip them)
                if gap >= 0.15:
                    _flush_sentence()
                    ET.SubElement(current_sent, "break", time=f"{int(gap * 1000)}ms")
                    current_sent = ET.SubElement(current_para, "s")

        sent_parts.append(word)

        # Punctuation-driven sentence split
        if any(word.endswith(p) for p in sentence_end_punct) and i < len(words) - 1:
            _flush_sentence()
            current_sent = ET.SubElement(current_para, "s")

    _flush_sentence()
    return _pretty_xml(speak)


def _pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(rough)
    # toprettyxml adds an XML declaration; remove it for cleaner output
    lines = reparsed.toprettyxml(indent="  ").splitlines()
    return "\n".join(line for line in lines if line.strip() and not line.startswith("<?xml"))


# ---------------------------------------------------------------------------
# Public dispatch helper
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    backend: str = "whisper",
    model: str = "base",
    bucket: str | None = None,
    profile: str | None = None,
    region: str = "us-east-1",
) -> list[dict]:
    """Transcribe *audio_path* and return a list of ``{word, start, end}`` dicts.

    Parameters
    ----------
    audio_path : str
        Path to a WAV file (use :func:`extract_audio` to produce one).
    backend : ``"whisper"`` | ``"aws-transcribe"``
        Which transcription engine to use.
    model : str
        Whisper model name (``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large"``).  Ignored for the AWS backend.
    bucket : str or None
        S3 bucket name required by the ``"aws-transcribe"`` backend.
    profile : str or None
        AWS profile name.  Ignored for the Whisper backend.
    region : str
        AWS region.  Ignored for the Whisper backend.
    """
    if backend == "aws-transcribe":
        if not bucket:
            raise ValueError("bucket is required for the aws-transcribe backend")
        return _transcribe_aws(audio_path, bucket, profile, region)
    return _transcribe_whisper(audio_path, model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a video and transcribe it to plain text and SSML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--backend",
        choices=["whisper", "aws-transcribe"],
        default="whisper",
        help="Transcription backend (default: whisper)",
    )
    # Whisper options
    parser.add_argument(
        "--model",
        default="base",
        metavar="MODEL",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    # AWS options
    parser.add_argument(
        "--bucket",
        metavar="S3_BUCKET",
        help="S3 bucket for temporary audio storage (aws-transcribe only)",
    )
    parser.add_argument(
        "--profile",
        metavar="AWS_PROFILE",
        help="AWS profile name (aws-transcribe only)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        metavar="AWS_REGION",
        help="AWS region (aws-transcribe only, default: us-east-1)",
    )
    # Output options
    parser.add_argument(
        "--text-output",
        metavar="PATH",
        help="Write plain-text transcript to this file (default: <video>.txt)",
    )
    parser.add_argument(
        "--ssml-output",
        metavar="PATH",
        help="Write SSML transcript to this file (default: <video>.ssml)",
    )
    parser.add_argument(
        "--audio-output",
        metavar="PATH",
        help="Keep the extracted WAV at this path (default: deleted after use)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print text and SSML to stdout instead of writing files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON file with per-word timestamps (<video>.words.json)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(2)

    if args.backend == "aws-transcribe" and not args.bucket:
        parser.error("--bucket is required when using --backend aws-transcribe")

    # --- Step 1: extract audio ---
    use_temp = args.audio_output is None
    audio_path = args.audio_output or str(
        Path(tempfile.mkdtemp()) / f"{video_path.stem}.wav"
    )
    try:
        print(f"Extracting audio from '{video_path}' …", file=sys.stderr)
        extract_audio(str(video_path), audio_path)
        print(f"Audio written to: {audio_path}", file=sys.stderr)

        # --- Step 2: transcribe ---
        if args.backend == "aws-transcribe":
            words = _transcribe_aws(audio_path, args.bucket, args.profile, args.region)
        else:
            words = _transcribe_whisper(audio_path, args.model)

    finally:
        if use_temp:
            Path(audio_path).unlink(missing_ok=True)

    if not words:
        print("Warning: transcription returned no words.", file=sys.stderr)

    # --- Step 3: generate outputs ---
    plain_text = words_to_text(words)
    ssml_text = words_to_ssml(words)

    if args.stdout:
        print("=== PLAIN TEXT ===")
        print(plain_text)
        print("\n=== SSML ===")
        print(ssml_text)
    else:
        text_path = Path(args.text_output) if args.text_output else video_path.with_suffix(".txt")
        ssml_path = Path(args.ssml_output) if args.ssml_output else video_path.with_suffix(".ssml")

        text_path.write_text(plain_text, encoding="utf-8")
        print(f"Text transcript written to: {text_path}", file=sys.stderr)

        ssml_path.write_text(ssml_text, encoding="utf-8")
        print(f"SSML transcript written to: {ssml_path}", file=sys.stderr)

    if args.json:
        json_path = video_path.with_suffix(".words.json")
        json_path.write_text(json.dumps(words, indent=2), encoding="utf-8")
        print(f"Word timestamps written to: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
