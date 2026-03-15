# protagonist

A system to build personalized videos by editing existing videos and including new audio with subtitles.

## Stack
- Textual 7.x for TUI
- External CSS in app.tcss (live reload via `textual run --dev`)
- Tests via Textual Pilot API + pytest
- Typer CLI in `app.py`

## Dev workflow
- Run: `textual run --dev app.py`
- Test: `pytest tests/`
- MCP: textual-mcp-server is configured in .mcp.json
- Use `.venv/bin/python app.py` (system Python lacks textual/typer)

## Architecture

### Entry point
`app.py` — Typer CLI + Textual TUI.  All pipeline commands live here.
`src/` modules are added to `sys.path` at import time.

### Pipeline workflow

1. Detect: Uses detector.py and text_analysis.py. Takes a video file as input and uses ifnude and/or AWS Rekognition to describe the video. Saves the combined output as a text file.
2. Transcribe: Uses transcribe.py. Takes a video file as input and uses Whisper or AWS Transcribe to transcribe audio and returns text or SSML. Outputs transcription.txt.
3. Regular expression matching: Uses regex_rules.py. Inputs regex.yaml. Allows a user to match specific transcriptions and/or SSML and create new snippets. The text snippets are used to create subtitles and new audio snippet. Outputs snippets.txt.
4. Subtitle: Uses create_subtitles.py. Input snippets.txt. Creates subtitles from the text snippets. Outputs subtitles.srt.
5. Polly: Uses polly.py. Input snippets.txt. Creates audio snippets from the text snippets. Outputs audio.mp3.
6. Combine: Uses a helper in app.py. Combine audio.mp3 and the video file.

### Pipeline modules (`src/`)
| Module | Key exports |
|---|---|
| `transcribe.py` | `extract_audio`, `transcribe_audio`, `words_to_text`, `words_to_ssml` |
| `detector.py` | `ifnude_detect_video`, `rekognition_detect_video`, `combine_results`, `EXPLICIT_CLASSES`, `ALL_CLASSES`, `VIDEO_EXTENSIONS` |
| `text_analysis.py` | `combine_with_detections`, `format_combined_analysis` |
| `create_subtitles.py` | `frames_to_subtitle_entries`, `generate_vtt`, `generate_srt` |
| `polly.py` | `synthesize_subtitles`, `POLLY_VOICES` |
| `regex_rules.py` | `RegexRule`, `apply_rules` |

`detector.py` has top-level imports of `boto3`, `cv2`, and `ifnude`; it is imported with a `try/except ImportError` guard so the CLI works without those deps.

### CLI commands
| Command | Description |
|---|---|
| `dashboard` | Launch the Textual TUI |
| `status` | Report dependency availability (ffmpeg, whisper, boto3, cv2, ifnude, pydub) |
| `transcribe` | Extract audio + transcribe → `.txt`, `.ssml`, optionally `.words.json` |
| `detect` | ifnude + Rekognition → `.detections.json` |
| `subtitles` | Analysis JSON → `.vtt` / `.srt` |
| `synthesize` | Analysis JSON → Polly MP3 → `.mp3` |
| `process` | Full pipeline (`--skip-detection`, `--skip-synthesis` flags available) |

### Output naming convention
All outputs are written next to the source file (or to `--output-dir`) using `<stem><suffix>`:
`.wav`, `.txt`, `.ssml`, `.words.json`, `.detections.json`, `.analysis.json`, `.vtt`, `.srt`, `.mp3`