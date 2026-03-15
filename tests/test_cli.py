"""
Tests for the Typer CLI commands in app.py.

All external I/O (ffmpeg, whisper, AWS, Polly) is mocked so tests run
without any installed media tools or cloud credentials.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app import app_cli

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.6, "end": 1.0},
]

FAKE_FRAMES = [
    {
        "frame": 0,
        "timestamp_sec": 0.0,
        "ifnude": [],
        "rekognition": [],
        "words_spoken": "Hello world",
        "matched_snippets": ["Test snippet"],
    }
]

FAKE_ANALYSIS = {
    "text_stats": {"word_count": 2, "unique_words": 2, "duration_sec": 1.0},
    "ssml_stats": {"paragraph_count": 1, "sentence_count": 1, "break_count": 0},
    "frames": FAKE_FRAMES,
}


@pytest.fixture
def video_file(tmp_path):
    """A placeholder .mp4 file (content doesn't matter — ffmpeg is mocked)."""
    f = tmp_path / "test.mp4"
    f.write_bytes(b"\x00" * 16)
    return f


@pytest.fixture
def analysis_file(tmp_path):
    """A pre-built analysis JSON file. Named 'sample.json' so stem='sample'."""
    f = tmp_path / "sample.json"
    f.write_text(json.dumps(FAKE_ANALYSIS), encoding="utf-8")
    return f


@pytest.fixture
def rules_file(tmp_path):
    """A minimal regex.yaml file."""
    f = tmp_path / "regex.yaml"
    f.write_text(
        "rules:\n"
        "  - name: test\n"
        "    transcript_pattern: ''\n"
        "    label_pattern: ''\n"
        "    snippet: 'Test snippet'\n",
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_runs():
    result = runner.invoke(app_cli, ["status"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.output


def test_status_shows_all_deps():
    result = runner.invoke(app_cli, ["status"])
    for dep in ("ffmpeg", "whisper", "boto3", "cv2", "ifnude", "pydub"):
        assert dep in result.output


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
def test_transcribe_writes_txt_and_ssml(mock_extract, mock_transcribe, video_file, tmp_path):
    result = runner.invoke(app_cli, [
        "transcribe", str(video_file),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "test.txt").exists()
    assert (tmp_path / "test.ssml").exists()


@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
def test_transcribe_writes_words_json_with_flag(mock_extract, mock_transcribe, video_file, tmp_path):
    result = runner.invoke(app_cli, [
        "transcribe", str(video_file),
        "--output-dir", str(tmp_path),
        "--json",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "test.words.json").exists()


@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
def test_transcribe_txt_content(mock_extract, mock_transcribe, video_file, tmp_path):
    runner.invoke(app_cli, [
        "transcribe", str(video_file),
        "--output-dir", str(tmp_path),
    ])
    txt = (tmp_path / "test.txt").read_text()
    assert "Hello" in txt
    assert "world" in txt


def test_transcribe_missing_video(tmp_path):
    result = runner.invoke(app_cli, ["transcribe", str(tmp_path / "missing.mp4")])
    assert result.exit_code != 0


def test_transcribe_aws_requires_bucket(video_file):
    result = runner.invoke(app_cli, [
        "transcribe", str(video_file),
        "--backend", "aws-transcribe",
    ])
    assert result.exit_code != 0
    assert "bucket" in result.output.lower()


@patch("app.extract_audio", side_effect=RuntimeError("ffmpeg not found"))
def test_transcribe_ffmpeg_error_exits(mock_extract, video_file, tmp_path):
    result = runner.invoke(app_cli, [
        "transcribe", str(video_file),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

@patch("app._detector_available", False)
def test_detect_missing_deps_exits(video_file):
    result = runner.invoke(app_cli, ["detect", str(video_file)])
    assert result.exit_code == 1
    assert "install" in result.stderr.lower()


def test_detect_missing_video(tmp_path):
    result = runner.invoke(app_cli, ["detect", str(tmp_path / "missing.mp4")])
    assert result.exit_code != 0


@patch("app._detector_available", True)
@patch("app.VIDEO_EXTENSIONS", {".mp4", ".mov"})
def test_detect_bad_extension(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"")
    result = runner.invoke(app_cli, ["detect", str(f)])
    assert result.exit_code != 0


@patch("app._detector_available", True)
@patch("app.VIDEO_EXTENSIONS", {".mp4"})
@patch("app.combine_results", return_value=[])
@patch("app.rekognition_detect_video", return_value=[])
@patch("app.ifnude_detect_video", return_value=[])
def test_detect_writes_detections_json(
    mock_ifnude, mock_rekog, mock_combine, video_file, tmp_path
):
    result = runner.invoke(app_cli, [
        "detect", str(video_file),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "test.detections.json").exists()


@patch("app._detector_available", True)
@patch("app.VIDEO_EXTENSIONS", {".mp4"})
@patch("app.combine_results", return_value=[])
@patch("app.rekognition_detect_video", return_value=[])
@patch("app.ifnude_detect_video", return_value=[])
def test_detect_skip_flags(mock_ifnude, mock_rekog, mock_combine, video_file, tmp_path):
    runner.invoke(app_cli, [
        "detect", str(video_file),
        "--output-dir", str(tmp_path),
        "--no-nudenet",
        "--no-rekognition",
    ])
    mock_ifnude.assert_not_called()
    mock_rekog.assert_not_called()


# ---------------------------------------------------------------------------
# subtitles
# ---------------------------------------------------------------------------

def test_subtitles_writes_vtt_and_srt(analysis_file, tmp_path):
    result = runner.invoke(app_cli, [
        "subtitles", str(analysis_file),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sample.vtt").exists()
    assert (tmp_path / "sample.srt").exists()


def test_subtitles_vtt_only(analysis_file, tmp_path):
    runner.invoke(app_cli, [
        "subtitles", str(analysis_file),
        "--output-dir", str(tmp_path),
        "--format", "vtt",
    ])
    assert (tmp_path / "sample.vtt").exists()
    assert not (tmp_path / "sample.srt").exists()


def test_subtitles_srt_only(analysis_file, tmp_path):
    runner.invoke(app_cli, [
        "subtitles", str(analysis_file),
        "--output-dir", str(tmp_path),
        "--format", "srt",
    ])
    assert (tmp_path / "sample.srt").exists()
    assert not (tmp_path / "sample.vtt").exists()


def test_subtitles_missing_file(tmp_path):
    result = runner.invoke(app_cli, ["subtitles", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


def test_subtitles_no_snippets_warns(tmp_path):
    empty_analysis = {
        "text_stats": {}, "ssml_stats": {}, "frames": [
            {"frame": 0, "timestamp_sec": 0.0, "ifnude": [], "rekognition": [],
             "words_spoken": "", "matched_snippets": []}
        ]
    }
    f = tmp_path / "empty.analysis.json"
    f.write_text(json.dumps(empty_analysis))
    result = runner.invoke(app_cli, [
        "subtitles", str(f),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "warning" in result.stderr.lower()


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

@patch("app.synthesize_subtitles")
def test_synthesize_writes_mp3(mock_synth, analysis_file, tmp_path):
    fake_mp3 = tmp_path / "_tmp.mp3"
    fake_mp3.write_bytes(b"")
    mock_synth.return_value = str(fake_mp3)

    result = runner.invoke(app_cli, [
        "synthesize", str(analysis_file),
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sample.mp3").exists()


@patch("app.synthesize_subtitles")
def test_synthesize_passes_voice(mock_synth, analysis_file, tmp_path):
    fake_mp3 = tmp_path / "_tmp.mp3"
    fake_mp3.write_bytes(b"")
    mock_synth.return_value = str(fake_mp3)

    runner.invoke(app_cli, [
        "synthesize", str(analysis_file),
        "--output-dir", str(tmp_path),
        "--voice", "Matthew",
    ])
    mock_synth.assert_called_once()
    _, kwargs = mock_synth.call_args
    assert kwargs.get("voice_id") == "Matthew"


def test_synthesize_missing_file(tmp_path):
    result = runner.invoke(app_cli, ["synthesize", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


def test_synthesize_no_snippets_exits(tmp_path):
    empty = {"frames": [
        {"frame": 0, "timestamp_sec": 0.0, "ifnude": [], "rekognition": [],
         "words_spoken": "", "matched_snippets": []}
    ]}
    f = tmp_path / "empty.analysis.json"
    f.write_text(json.dumps(empty))
    result = runner.invoke(app_cli, ["synthesize", str(f)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

def _patch_process():
    """Return a list of context manager patches for the full process pipeline."""
    return [
        patch("app.extract_audio"),
        patch("app.transcribe_audio", return_value=FAKE_WORDS),
        patch("app.combine_with_detections", return_value=FAKE_ANALYSIS),
        patch("app.synthesize_subtitles", return_value="/tmp/fake.mp3"),
        patch("app._combine_video_audio"),
        patch("app._detector_available", False),
        patch("pathlib.Path.write_text"),
    ]


@patch("app._combine_video_audio")
@patch("app.synthesize_subtitles", return_value="/tmp/fake.mp3")
@patch("app.combine_with_detections", return_value=FAKE_ANALYSIS)
@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
@patch("app._detector_available", False)
def test_process_skip_detection_skip_synthesis(
    mock_extract, mock_transcribe, mock_combine, mock_synth, mock_ffmpeg,
    video_file, tmp_path,
):
    result = runner.invoke(app_cli, [
        "process", str(video_file),
        "--output-dir", str(tmp_path),
        "--skip-detection",
        "--skip-synthesis",
    ])
    assert result.exit_code == 0, result.output
    mock_synth.assert_not_called()
    mock_ffmpeg.assert_not_called()


@patch("app._combine_video_audio")
@patch("app.synthesize_subtitles")
@patch("app.combine_with_detections", return_value=FAKE_ANALYSIS)
@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
@patch("app._detector_available", False)
def test_process_runs_all_steps(
    mock_extract, mock_transcribe, mock_combine, mock_synth, mock_ffmpeg,
    video_file, tmp_path,
):
    import tempfile as _tf
    fake_mp3 = Path(_tf.mktemp(suffix=".mp3"))
    fake_mp3.write_bytes(b"")
    mock_synth.return_value = str(fake_mp3)

    result = runner.invoke(app_cli, [
        "process", str(video_file),
        "--output-dir", str(tmp_path),
        "--skip-detection",
    ])
    assert result.exit_code == 0, result.output
    mock_extract.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_combine.assert_called_once()
    mock_ffmpeg.assert_called_once()


@patch("app._detector_available", False)
def test_process_missing_video(tmp_path):
    result = runner.invoke(app_cli, [
        "process", str(tmp_path / "missing.mp4"),
        "--skip-detection", "--skip-synthesis",
    ])
    assert result.exit_code != 0


@patch("app._combine_video_audio")
@patch("app.synthesize_subtitles")
@patch("app.combine_with_detections", return_value=FAKE_ANALYSIS)
@patch("app.transcribe_audio", return_value=FAKE_WORDS)
@patch("app.extract_audio")
@patch("app._detector_available", False)
def test_process_loads_rules_file(
    mock_extract, mock_transcribe, mock_combine, mock_synth, mock_ffmpeg,
    video_file, rules_file, tmp_path,
):
    import tempfile as _tf
    fake_mp3 = Path(_tf.mktemp(suffix=".mp3"))
    fake_mp3.write_bytes(b"")
    mock_synth.return_value = str(fake_mp3)

    runner.invoke(app_cli, [
        "process", str(video_file),
        "--rules", str(rules_file),
        "--output-dir", str(tmp_path),
        "--skip-detection",
    ])
    _, kwargs = mock_combine.call_args
    assert kwargs.get("rules") is not None
    assert len(kwargs["rules"]) == 1
    assert kwargs["rules"][0].snippet == "Test snippet"


def test_process_aws_transcribe_requires_bucket(video_file):
    result = runner.invoke(app_cli, [
        "process", str(video_file),
        "--backend", "aws-transcribe",
        "--skip-detection", "--skip-synthesis",
    ])
    assert result.exit_code != 0
    assert "bucket" in result.output.lower()
