#!/usr/bin/env python3
"""
Generate subtitle files (WebVTT and SRT) from combined analysis frames
that have matched regex rule snippets.

Public API
----------
frames_to_subtitle_entries(frames, min_duration)
    Convert frames with matched_snippets into timed subtitle entries.

generate_vtt(frames)
    Return a WebVTT string from analysis frames.

generate_srt(frames)
    Return an SRT string from analysis frames.

save_vtt(frames)
    Write a WebVTT file to a temp path and return that path.

save_srt(frames)
    Write an SRT file to a temp path and return that path.
"""

import tempfile


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = round((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def frames_to_subtitle_entries(
    frames: list[dict],
    min_duration: float = 2.0,
) -> list[dict]:
    """Convert frames with matched_snippets to timed subtitle entries.

    Only frames that have at least one matched snippet and a valid
    ``timestamp_sec`` are included.  Consecutive frames with identical
    snippet text are merged into a single entry to reduce flickering.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`,
        each containing ``timestamp_sec`` and ``matched_snippets``.
    min_duration : float
        Minimum subtitle display duration in seconds (default 2.0).

    Returns
    -------
    list[dict]
        Sorted list of ``{start, end, text}`` dicts ready for formatting.
    """
    snippet_frames = [
        f for f in frames
        if f.get("matched_snippets") and f.get("timestamp_sec") is not None
    ]
    if not snippet_frames:
        return []

    # Build raw entries, then merge adjacent entries with the same text.
    raw: list[dict] = []
    for i, frame in enumerate(snippet_frames):
        start = float(frame["timestamp_sec"])
        if i + 1 < len(snippet_frames):
            next_start = float(snippet_frames[i + 1]["timestamp_sec"])
            end = min(start + min_duration, next_start - 0.05)
        else:
            end = start + min_duration
        end = max(end, start + 0.5)  # guarantee a minimum display window
        text = " | ".join(frame["matched_snippets"])
        raw.append({"start": start, "end": end, "text": text})

    # Merge consecutive entries that share the same text.
    merged: list[dict] = [raw[0].copy()]
    for entry in raw[1:]:
        if entry["text"] == merged[-1]["text"]:
            merged[-1]["end"] = entry["end"]
        else:
            merged.append(entry.copy())

    return merged


def generate_vtt(frames: list[dict]) -> str:
    """Generate a WebVTT subtitle string from analysis frames.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`.

    Returns
    -------
    str
        A complete WebVTT document (may contain zero cue blocks if no
        frames have matched snippets).
    """
    entries = frames_to_subtitle_entries(frames)
    lines = ["WEBVTT", ""]
    for i, entry in enumerate(entries, 1):
        start = _format_vtt_time(entry["start"])
        end = _format_vtt_time(entry["end"])
        lines.extend([str(i), f"{start} --> {end}", entry["text"], ""])
    return "\n".join(lines)


def generate_srt(frames: list[dict]) -> str:
    """Generate an SRT subtitle string from analysis frames.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`.

    Returns
    -------
    str
        A complete SRT document (empty string if no frames have matched
        snippets).
    """
    entries = frames_to_subtitle_entries(frames)
    lines = []
    for i, entry in enumerate(entries, 1):
        start = _format_srt_time(entry["start"])
        end = _format_srt_time(entry["end"])
        lines.extend([str(i), f"{start} --> {end}", entry["text"], ""])
    return "\n".join(lines)


def save_vtt(frames: list[dict]) -> str:
    """Write WebVTT subtitles to a temporary file and return its path.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`.

    Returns
    -------
    str
        Absolute path to the temporary ``.vtt`` file.
    """
    content = generate_vtt(frames)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".vtt", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def save_srt(frames: list[dict]) -> str:
    """Write SRT subtitles to a temporary file and return its path.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`.

    Returns
    -------
    str
        Absolute path to the temporary ``.srt`` file.
    """
    content = generate_srt(frames)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".srt", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name
