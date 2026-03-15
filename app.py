import sys
import shutil
import json
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer
from enum import Enum
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Footer, Header, Input, Label,
    ListItem, ListView, RichLog, Select, Static,
)

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transcribe import extract_audio, transcribe_audio, words_to_text, words_to_ssml

try:
    from detector import (
        ifnude_detect_video,
        rekognition_detect_video,
        combine_results,
        EXPLICIT_CLASSES,
        ALL_CLASSES,
        VIDEO_EXTENSIONS,
    )
    _detector_available = True
except ImportError:
    _detector_available = False

from text_analysis import combine_with_detections
from create_subtitles import frames_to_subtitle_entries, generate_vtt, generate_srt
from polly import synthesize_subtitles, POLLY_VOICES
from regex_rules import RegexRule

app_cli = typer.Typer()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _out_path(source: Path, suffix: str, output_dir: Optional[Path]) -> Path:
    base = output_dir if output_dir else source.parent
    return base / f"{source.stem}{suffix}"


def _import_check(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _load_rules(rules_file: Path) -> list[RegexRule]:
    """Parse a regex.yaml file into a list of RegexRule objects.

    Expected YAML structure::

        rules:
          - name: "greeting"
            transcript_pattern: "hello|hi"
            label_pattern: ""
            snippet: "Hello there!"
    """
    try:
        import yaml
    except ImportError:
        typer.echo("pyyaml is required to load rules.  Install with: pip install pyyaml", err=True)
        raise typer.Exit(1)

    data = yaml.safe_load(rules_file.read_text(encoding="utf-8")) or {}
    rules = []
    for entry in data.get("rules", []):
        rules.append(RegexRule(
            name=entry.get("name", ""),
            transcript_pattern=entry.get("transcript_pattern", ""),
            label_pattern=entry.get("label_pattern", ""),
            snippet=entry.get("snippet", ""),
        ))
    return rules


def _combine_video_audio(video: Path, audio: Path, output: Path, srt: Optional[Path] = None) -> None:
    """Merge an MP3 audio track onto a video using ffmpeg, burning in SRT subtitles if provided."""
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio)]
    if srt:
        # Escape colons and backslashes in the path for the subtitles filter
        srt_escaped = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
        cmd += ["-filter_complex", f"[0:v]subtitles={srt_escaped}[v]", "-map", "[v]"]
    else:
        cmd += ["-map", "0:v", "-c:v", "copy"]
    cmd += ["-map", "1:a", "-shortest", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg combine failed:\n{result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Shared option type aliases
# ---------------------------------------------------------------------------

ProfileOpt   = Annotated[Optional[str],  typer.Option("--profile",    help="AWS profile")]
RegionOpt    = Annotated[str,            typer.Option("--region",      help="AWS region")]
BucketOpt    = Annotated[Optional[str],  typer.Option("--bucket",      help="S3 bucket (aws-transcribe only)")]
OutputDirOpt = Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Output directory")]


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"

_STATUS_STYLE = {
    StepStatus.PENDING: ("[ ]", "dim"),
    StepStatus.RUNNING: ("[~]", "bold yellow"),
    StepStatus.DONE:    ("[✓]", "bold green"),
    StepStatus.ERROR:   ("[✗]", "bold red"),
}

_PIPELINE_STEPS = [
    "1. Detect",
    "2. Transcribe",
    "3. Regex Matching",
    "4. Subtitles",
    "5. Polly Synthesis",
    "6. Combine",
]

_BACKENDS = [("Whisper (local)", "whisper"), ("AWS Transcribe", "aws-transcribe")]
_VOICES   = [(v, v) for v in ["Joanna", "Matthew", "Amy", "Brian", "Emma", "Russell"]]


class PipelineStepItem(ListItem):
    """Single pipeline step row with live status indicator."""

    status: reactive[StepStatus] = reactive(StepStatus.PENDING)

    def __init__(self, step_name: str, step_index: int) -> None:
        super().__init__(id=f"step-{step_index}")
        self.step_name = step_name
        self.step_index = step_index

    def compose(self) -> ComposeResult:
        icon, _ = _STATUS_STYLE[StepStatus.PENDING]
        yield Label(f"{icon} {self.step_name}", id=f"step-label-{self.step_index}")

    def watch_status(self, status: StepStatus) -> None:
        icon, style = _STATUS_STYLE[status]
        self.query_one(Label).update(f"[{style}]{icon} {self.step_name}[/]")


class Protagonist(App):
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+r", "run_pipeline",  "Run",  show=True),
        Binding("ctrl+c", "quit",          "Quit", show=True),
    ]

    # Messages -----------------------------------------------------------

    class StepStarted(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    class StepDone(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    class StepFailed(Message):
        def __init__(self, index: int, error: str) -> None:
            self.index = index
            self.error = error
            super().__init__()

    # Layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Static("Video Pipeline", id="panel-title")
                yield Input(placeholder="Video path...", id="video-path")
                yield Input(placeholder="Rules file (regex.yaml)...", id="rules-path")
                yield Select(_BACKENDS, prompt="Backend", id="backend-select", value="whisper")
                yield Select(_VOICES,   prompt="Polly voice", id="voice-select", value="Joanna")
                yield Input(placeholder="Output dir (optional)...", id="output-dir")
                with Horizontal(id="btn-row"):
                    yield Button("▶  Run", id="btn-run", variant="success")
                    yield Button("■  Stop", id="btn-stop", variant="error")
                yield ListView(
                    *[PipelineStepItem(name, i) for i, name in enumerate(_PIPELINE_STEPS)],
                    id="step-list",
                )
            with Vertical(id="right-panel"):
                yield Static("Output Log", id="log-title")
                yield RichLog(highlight=True, markup=True, id="log", max_lines=2000, wrap=True)
        yield Footer()

    # Event handlers -----------------------------------------------------

    @on(Button.Pressed, "#btn-run")
    def on_run(self, _: Button.Pressed) -> None:
        self.action_run_pipeline()

    @on(Button.Pressed, "#btn-stop")
    def on_stop(self, _: Button.Pressed) -> None:
        self.workers.cancel_all()
        self._log("[yellow]Pipeline stopped by user.[/]")

    @on(StepStarted)
    def handle_started(self, msg: "Protagonist.StepStarted") -> None:
        self.query_one(f"#step-{msg.index}", PipelineStepItem).status = StepStatus.RUNNING

    @on(StepDone)
    def handle_done(self, msg: "Protagonist.StepDone") -> None:
        self.query_one(f"#step-{msg.index}", PipelineStepItem).status = StepStatus.DONE

    @on(StepFailed)
    def handle_failed(self, msg: "Protagonist.StepFailed") -> None:
        self.query_one(f"#step-{msg.index}", PipelineStepItem).status = StepStatus.ERROR
        self._log(f"[bold red]Step failed: {msg.error}[/]")

    # Actions ------------------------------------------------------------

    def action_run_pipeline(self) -> None:
        video_val   = self.query_one("#video-path",    Input).value.strip()
        rules_val   = self.query_one("#rules-path",    Input).value.strip()
        output_val  = self.query_one("#output-dir",    Input).value.strip()
        backend_val = self.query_one("#backend-select", Select).value
        voice_val   = self.query_one("#voice-select",   Select).value

        if backend_val is Select.BLANK:
            backend_val = "whisper"
        if voice_val is Select.BLANK:
            voice_val = "Joanna"

        if not video_val:
            self._log("[red]Error: video path is required.[/]")
            return

        # Reset step statuses
        for i in range(len(_PIPELINE_STEPS)):
            self.query_one(f"#step-{i}", PipelineStepItem).status = StepStatus.PENDING

        self.query_one("#log", RichLog).clear()
        self._run_pipeline_worker(
            video=video_val,
            rules=rules_val or None,
            backend=str(backend_val),
            voice=str(voice_val),
            output_dir=output_val or None,
        )

    # Worker -------------------------------------------------------------

    @work(thread=True)
    def _run_pipeline_worker(
        self,
        video: str,
        rules: Optional[str],
        backend: str,
        voice: str,
        output_dir: Optional[str],
    ) -> None:
        video_path  = Path(video)
        rules_path  = Path(rules) if rules else None
        out_dir     = Path(output_dir) if output_dir else None

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        loaded_rules = _load_rules(rules_path) if rules_path else []

        def start(i: int) -> None:
            self.app.post_message(Protagonist.StepStarted(i))

        def done(i: int) -> None:
            self.app.post_message(Protagonist.StepDone(i))

        def fail(i: int, err: str) -> None:
            self.app.post_message(Protagonist.StepFailed(i, err))

        def log(msg: str) -> None:
            self.call_from_thread(self.query_one("#log", RichLog).write, msg)

        # [1] Detect
        start(0)
        combined = []
        if not _detector_available:
            log("[yellow]Detector deps missing — skipping detection.[/]")
            done(0)
        else:
            try:
                log("Running content detection...")
                classes = EXPLICIT_CLASSES
                ifnude_r, rekog_r = [], []
                ifnude_r   = ifnude_detect_video(str(video_path), 0.5, classes, 30)
                rekog_r    = rekognition_detect_video(str(video_path), 0.5, None, 30)
                combined   = combine_results(ifnude_r, rekog_r, is_video=True)
                det_path   = _out_path(video_path, ".detections.json", out_dir)
                det_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
                log(f"  -> {det_path}")
                done(0)
            except Exception as e:
                fail(0, str(e))
                return

        # [2] Transcribe
        start(1)
        tmp_dir    = tempfile.mkdtemp()
        audio_path = str(Path(tmp_dir) / f"{video_path.stem}.wav")
        words      = []
        try:
            log(f"Extracting audio and transcribing with '{backend}'...")
            extract_audio(str(video_path), audio_path)
            words = transcribe_audio(audio_path, backend=backend)
            txt_path = _out_path(video_path, ".txt", out_dir)
            txt_path.write_text(words_to_text(words), encoding="utf-8")
            log(f"  -> {txt_path}")
            done(1)
        except Exception as e:
            fail(1, str(e))
            return
        finally:
            Path(audio_path).unlink(missing_ok=True)

        # [3] Regex matching
        start(2)
        try:
            log("Applying regex rules...")
            analysis     = combine_with_detections(combined, words, is_video=True, rules=loaded_rules)
            analysis_path = _out_path(video_path, ".analysis.json", out_dir)
            analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            frames   = analysis.get("frames", [])
            entries  = frames_to_subtitle_entries(frames)
            snip_path = _out_path(video_path, ".snippets.txt", out_dir)
            snip_path.write_text("\n".join(e["text"] for e in entries), encoding="utf-8")
            log(f"  -> {analysis_path}, {snip_path} ({len(entries)} snippet(s))")
            done(2)
        except Exception as e:
            fail(2, str(e))
            return

        # [4] Subtitles
        start(3)
        try:
            log("Generating subtitles...")
            srt_path = _out_path(video_path, ".srt", out_dir)
            vtt_path = _out_path(video_path, ".vtt", out_dir)
            srt_path.write_text(generate_srt(frames), encoding="utf-8")
            vtt_path.write_text(generate_vtt(frames), encoding="utf-8")
            log(f"  -> {srt_path}, {vtt_path}")
            done(3)
        except Exception as e:
            fail(3, str(e))
            return

        # [5] Polly synthesis
        start(4)
        mp3_path = None
        if not entries:
            log("[yellow]No snippets — skipping Polly synthesis.[/]")
            done(4)
        else:
            try:
                log(f"Synthesizing audio with Polly voice '{voice}'...")
                tmp_mp3  = synthesize_subtitles(entries, voice_id=voice)
                mp3_path = _out_path(video_path, ".mp3", out_dir)
                shutil.move(tmp_mp3, str(mp3_path))
                log(f"  -> {mp3_path}")
                done(4)
            except Exception as e:
                fail(4, str(e))
                return

        # [6] Combine
        start(5)
        if not mp3_path:
            log("[yellow]No audio — skipping combine.[/]")
            done(5)
        else:
            try:
                log("Combining video with audio and subtitles...")
                out_video = _out_path(video_path, ".output" + video_path.suffix, out_dir)
                _combine_video_audio(video_path, mp3_path, out_video, srt=srt_path)
                log(f"  -> {out_video}")
                done(5)
            except Exception as e:
                fail(5, str(e))
                return

        log("[bold green]Pipeline complete![/]")

    # Helper -------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)


@app_cli.command()
def dashboard(verbose: bool = False):
    """Launch the TUI dashboard."""
    Protagonist().run()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app_cli.command()
def status():
    """Check and report availability of required dependencies."""
    deps = [
        ("ffmpeg",   shutil.which("ffmpeg") is not None),
        ("whisper",  _import_check("whisper")),
        ("boto3",    _import_check("boto3")),
        ("cv2",      _import_check("cv2")),
        ("ifnude",   _import_check("ifnude")),
        ("pydub",    _import_check("pydub")),
    ]
    for name, available in deps:
        mark = "OK" if available else "MISSING"
        typer.echo(f"  {name:<12} {mark}")


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

@app_cli.command()
def transcribe(
    video: Annotated[Path, typer.Argument(help="Input video file")],
    backend: Annotated[str, typer.Option("--backend", help="whisper or aws-transcribe")] = "whisper",
    model: Annotated[str, typer.Option("--model", help="Whisper model: tiny/base/small/medium/large")] = "base",
    bucket: BucketOpt = None,
    profile: ProfileOpt = None,
    region: RegionOpt = "us-east-1",
    output_dir: OutputDirOpt = None,
    keep_audio: Annotated[bool, typer.Option("--keep-audio", help="Keep extracted WAV")] = False,
    json_words: Annotated[bool, typer.Option("--json", help="Write per-word timestamps JSON")] = False,
):
    """Extract audio and transcribe a video file."""
    if not video.exists():
        raise typer.BadParameter(f"File not found: {video}", param_hint="video")

    if backend == "aws-transcribe" and not bucket:
        raise typer.BadParameter("--bucket is required for aws-transcribe backend", param_hint="--bucket")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp()
    audio_path = str(Path(tmp_dir) / f"{video.stem}.wav") if not keep_audio else str(_out_path(video, ".wav", output_dir))

    try:
        typer.echo(f"Extracting audio from '{video}' ...")
        try:
            extract_audio(str(video), audio_path)
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

        typer.echo(f"Transcribing with backend '{backend}' ...")
        try:
            words = transcribe_audio(audio_path, backend=backend, model=model, bucket=bucket, profile=profile, region=region)
        except Exception as e:
            _handle_aws_error(e)

    finally:
        if not keep_audio:
            Path(audio_path).unlink(missing_ok=True)

    txt_path = _out_path(video, ".txt", output_dir)
    ssml_path = _out_path(video, ".ssml", output_dir)

    txt_path.write_text(words_to_text(words), encoding="utf-8")
    typer.echo(f"Text written to: {txt_path}")

    ssml_path.write_text(words_to_ssml(words), encoding="utf-8")
    typer.echo(f"SSML written to: {ssml_path}")

    if json_words:
        words_path = _out_path(video, ".words.json", output_dir)
        words_path.write_text(json.dumps(words, indent=2), encoding="utf-8")
        typer.echo(f"Word timestamps written to: {words_path}")


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

@app_cli.command()
def detect(
    video: Annotated[Path, typer.Argument(help="Input video file")],
    profile: ProfileOpt = None,
    frame_interval: Annotated[int, typer.Option("--frame-interval", help="Analyse every Nth frame")] = 30,
    min_confidence: Annotated[float, typer.Option("--min-confidence", help="Minimum detection confidence")] = 0.5,
    all_classes: Annotated[bool, typer.Option("--all-classes", help="Detect all body-part classes")] = False,
    no_nudenet: Annotated[bool, typer.Option("--no-nudenet", help="Skip NudeNet/ifnude detection")] = False,
    no_rekognition: Annotated[bool, typer.Option("--no-rekognition", help="Skip Rekognition detection")] = False,
    output_dir: OutputDirOpt = None,
):
    """Detect content in a video and write detections JSON."""
    if not _detector_available:
        typer.echo(
            "detector dependencies are not available.  Install with:\n"
            "  pip install ifnude opencv-python boto3",
            err=True,
        )
        raise typer.Exit(1)

    if not video.exists():
        raise typer.BadParameter(f"File not found: {video}", param_hint="video")

    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise typer.BadParameter(
            f"Unsupported video extension '{video.suffix}'.  Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}",
            param_hint="video",
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    classes = ALL_CLASSES if all_classes else EXPLICIT_CLASSES
    ifnude_results = []
    rekognition_results = []

    try:
        if not no_nudenet:
            typer.echo("Running ifnude detection ...")
            ifnude_results = ifnude_detect_video(str(video), min_confidence, classes, frame_interval)

        if not no_rekognition:
            typer.echo("Running Rekognition detection ...")
            rekognition_results = rekognition_detect_video(str(video), min_confidence, profile, frame_interval)

    except Exception as e:
        _handle_aws_error(e)

    combined = combine_results(ifnude_results, rekognition_results, is_video=True)

    det_path = _out_path(video, ".detections.json", output_dir)
    det_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    typer.echo(f"Detections written to: {det_path}")


# ---------------------------------------------------------------------------
# subtitles
# ---------------------------------------------------------------------------

@app_cli.command()
def subtitles(
    analysis_file: Annotated[Path, typer.Argument(help="Path to .analysis.json file")],
    output_dir: OutputDirOpt = None,
    min_duration: Annotated[float, typer.Option("--min-duration", help="Minimum subtitle display duration (seconds)")] = 2.0,
    fmt: Annotated[str, typer.Option("--format", help="Output format: vtt, srt, or both")] = "both",
):
    """Generate subtitle files from a combined analysis JSON."""
    if not analysis_file.exists():
        raise typer.BadParameter(f"File not found: {analysis_file}", param_hint="analysis_file")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    frames = analysis.get("frames", [])

    entries = frames_to_subtitle_entries(frames, min_duration=min_duration)
    if not entries:
        typer.echo("Warning: no subtitle entries generated (no regex rules matched any frames).", err=True)

    if fmt in ("vtt", "both"):
        vtt_path = _out_path(analysis_file, ".vtt", output_dir)
        vtt_path.write_text(generate_vtt(frames), encoding="utf-8")
        typer.echo(f"VTT written to: {vtt_path}")

    if fmt in ("srt", "both"):
        srt_path = _out_path(analysis_file, ".srt", output_dir)
        srt_path.write_text(generate_srt(frames), encoding="utf-8")
        typer.echo(f"SRT written to: {srt_path}")


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

@app_cli.command()
def synthesize(
    analysis_file: Annotated[Path, typer.Argument(help="Path to .analysis.json file")],
    voice: Annotated[str, typer.Option("--voice", help=f"Polly voice ID (e.g. Joanna, Matthew)")] = "Joanna",
    profile: ProfileOpt = None,
    region: RegionOpt = "us-east-1",
    output_dir: OutputDirOpt = None,
):
    """Synthesize subtitle audio via AWS Polly."""
    if not analysis_file.exists():
        raise typer.BadParameter(f"File not found: {analysis_file}", param_hint="analysis_file")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    frames = analysis.get("frames", [])

    entries = frames_to_subtitle_entries(frames)
    if not entries:
        typer.echo("No subtitle entries to synthesize (no regex rules matched).", err=True)
        raise typer.Exit(1)

    try:
        tmp_mp3 = synthesize_subtitles(entries, voice_id=voice, profile=profile, region=region)
    except Exception as e:
        _handle_aws_error(e)

    mp3_path = _out_path(analysis_file, ".mp3", output_dir)
    shutil.move(tmp_mp3, str(mp3_path))
    typer.echo(f"Audio written to: {mp3_path}")


# ---------------------------------------------------------------------------
# process  (full pipeline)
# ---------------------------------------------------------------------------

@app_cli.command()
def process(
    video: Annotated[Path, typer.Argument(help="Input video file")],
    # regex rules
    rules_file: Annotated[Optional[Path], typer.Option("--rules", help="Path to regex.yaml rules file")] = None,
    # transcription options
    backend: Annotated[str, typer.Option("--backend", help="whisper or aws-transcribe")] = "whisper",
    model: Annotated[str, typer.Option("--model", help="Whisper model size")] = "base",
    bucket: BucketOpt = None,
    profile: ProfileOpt = None,
    region: RegionOpt = "us-east-1",
    # detection options
    frame_interval: Annotated[int, typer.Option("--frame-interval", help="Analyse every Nth frame")] = 30,
    min_confidence: Annotated[float, typer.Option("--min-confidence", help="Minimum detection confidence")] = 0.5,
    all_classes: Annotated[bool, typer.Option("--all-classes", help="Detect all body-part classes")] = False,
    no_nudenet: Annotated[bool, typer.Option("--no-nudenet", help="Skip NudeNet/ifnude detection")] = False,
    no_rekognition: Annotated[bool, typer.Option("--no-rekognition", help="Skip Rekognition detection")] = False,
    # synthesis options
    voice: Annotated[str, typer.Option("--voice", help="Polly voice ID")] = "Joanna",
    # pipeline control
    skip_detection: Annotated[bool, typer.Option("--skip-detection", help="Skip content detection step")] = False,
    skip_synthesis: Annotated[bool, typer.Option("--skip-synthesis", help="Skip Polly synthesis and combine steps")] = False,
    output_dir: OutputDirOpt = None,
):
    """Run the full protagonist pipeline end-to-end.

    Steps: detect → transcribe → regex matching → subtitles → polly → combine
    """
    if not video.exists():
        raise typer.BadParameter(f"File not found: {video}", param_hint="video")

    if backend == "aws-transcribe" and not bucket:
        raise typer.BadParameter("--bucket is required for aws-transcribe backend", param_hint="--bucket")

    if rules_file and not rules_file.exists():
        raise typer.BadParameter(f"Rules file not found: {rules_file}", param_hint="--rules")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rules = _load_rules(rules_file) if rules_file else []

    # --- [1/6] Detect content ---
    if skip_detection:
        typer.echo("[1/6] Skipping detection (--skip-detection).")
        combined = []
    else:
        if not _detector_available:
            typer.echo(
                "[1/6] detector dependencies missing; skipping.  "
                "Install with: pip install ifnude opencv-python boto3",
                err=True,
            )
            combined = []
        else:
            typer.echo("[1/6] Detecting content ...")
            classes = ALL_CLASSES if all_classes else EXPLICIT_CLASSES
            ifnude_results = []
            rekognition_results = []
            try:
                if not no_nudenet:
                    ifnude_results = ifnude_detect_video(str(video), min_confidence, classes, frame_interval)
                if not no_rekognition:
                    rekognition_results = rekognition_detect_video(str(video), min_confidence, profile, frame_interval)
            except Exception as e:
                _handle_aws_error(e)

            combined = combine_results(ifnude_results, rekognition_results, is_video=True)
            det_path = _out_path(video, ".detections.json", output_dir)
            det_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            typer.echo(f"  -> {det_path}")

    # --- [2/6] Transcribe ---
    typer.echo(f"[2/6] Transcribing with '{backend}' ...")
    tmp_dir = tempfile.mkdtemp()
    audio_path = str(Path(tmp_dir) / f"{video.stem}.wav")

    try:
        try:
            extract_audio(str(video), audio_path)
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

        try:
            words = transcribe_audio(audio_path, backend=backend, model=model, bucket=bucket, profile=profile, region=region)
        except Exception as e:
            _handle_aws_error(e)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    txt_path = _out_path(video, ".txt", output_dir)
    txt_path.write_text(words_to_text(words), encoding="utf-8")
    typer.echo(f"  -> {txt_path}")

    # --- [3/6] Regex matching ---
    typer.echo("[3/6] Applying regex rules ...")
    analysis = combine_with_detections(combined, words, is_video=True, rules=rules)
    analysis_path = _out_path(video, ".analysis.json", output_dir)
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    frames = analysis.get("frames", [])
    entries = frames_to_subtitle_entries(frames)
    snippets_path = _out_path(video, ".snippets.txt", output_dir)
    snippets_path.write_text(
        "\n".join(e["text"] for e in entries),
        encoding="utf-8",
    )
    typer.echo(f"  -> {analysis_path}, {snippets_path} ({len(entries)} snippet(s))")

    if not entries:
        typer.echo("  Warning: no rules matched; subtitles and audio will be empty.", err=True)

    # --- [4/6] Subtitles ---
    typer.echo("[4/6] Generating subtitles ...")
    srt_path = _out_path(video, ".srt", output_dir)
    vtt_path = _out_path(video, ".vtt", output_dir)
    srt_path.write_text(generate_srt(frames), encoding="utf-8")
    vtt_path.write_text(generate_vtt(frames), encoding="utf-8")
    typer.echo(f"  -> {srt_path}, {vtt_path}")

    # --- [5/6] Polly synthesis ---
    if skip_synthesis:
        typer.echo("[5/6] Skipping Polly synthesis (--skip-synthesis).")
        typer.echo("[6/6] Skipping combine (--skip-synthesis).")
        return

    if not entries:
        typer.echo("[5/6] No snippets to synthesize; skipping Polly and combine.", err=True)
        return

    typer.echo("[5/6] Synthesizing audio with Polly ...")
    try:
        tmp_mp3 = synthesize_subtitles(entries, voice_id=voice, profile=profile, region=region)
    except Exception as e:
        _handle_aws_error(e)

    mp3_path = _out_path(video, ".mp3", output_dir)
    shutil.move(tmp_mp3, str(mp3_path))
    typer.echo(f"  -> {mp3_path}")

    # --- [6/6] Combine video + audio + subtitles ---
    typer.echo("[6/6] Combining video with synthesized audio and subtitles ...")
    output_video = _out_path(video, ".output" + video.suffix, output_dir)
    try:
        _combine_video_audio(video, mp3_path, output_video, srt=srt_path)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"  -> {output_video}")


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _handle_aws_error(e: Exception) -> None:
    """Print a friendly message for common AWS errors and exit."""
    try:
        import botocore.exceptions
        if isinstance(e, botocore.exceptions.NoCredentialsError):
            typer.echo("AWS credentials not found.  Configure via 'aws configure' or environment variables.", err=True)
            raise typer.Exit(1)
    except ImportError:
        pass
    typer.echo(str(e), err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app_cli()
