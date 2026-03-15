#!/usr/bin/env python3
"""
Analyze transcribed text/SSML and combine it with detector output.

Public API
----------
analyze_text(words)
    Compute statistics from a list of timed word dicts.

analyze_ssml(ssml)
    Parse an SSML string and return structural statistics.

combine_with_detections(combined_detections, words, is_video, window_sec)
    Merge detector output with transcript, correlating by timestamp for video.

format_combined_analysis(analysis, is_video)
    Render the combined structure as human-readable text.
"""

import json
import xml.etree.ElementTree as ET

from transcribe import words_to_ssml, words_to_text
from regex_rules import RegexRule, apply_rules


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def analyze_text(words: list[dict]) -> dict:
    """Return statistics about a timed word list.

    Parameters
    ----------
    words : list[dict]
        Each dict has ``{"word": str, "start": float, "end": float}``.

    Returns
    -------
    dict with keys:
        word_count     – total number of word tokens
        unique_words   – number of distinct words (case-insensitive, stripped)
        duration_sec   – time from first word start to last word end (0 if empty)
        text           – plain-text string assembled from the words
    """
    if not words:
        return {"word_count": 0, "unique_words": 0, "duration_sec": 0.0, "text": ""}

    text = words_to_text(words)
    unique = {w["word"].lower().strip(".,!?;:—–") for w in words if w["word"].strip(".,!?;:—–")}
    duration = round(words[-1]["end"] - words[0]["start"], 2)

    return {
        "word_count": len(words),
        "unique_words": len(unique),
        "duration_sec": duration,
        "text": text,
    }


# ---------------------------------------------------------------------------
# SSML analysis
# ---------------------------------------------------------------------------

def analyze_ssml(ssml: str) -> dict:
    """Parse an SSML document and return structural statistics.

    Parameters
    ----------
    ssml : str
        A well-formed SSML XML string produced by :func:`transcribe.words_to_ssml`.

    Returns
    -------
    dict with keys:
        paragraph_count – number of ``<p>`` elements
        sentence_count  – number of ``<s>`` elements
        break_count     – number of ``<break>`` elements
    """
    try:
        root = ET.fromstring(ssml)
    except ET.ParseError:
        return {"paragraph_count": 0, "sentence_count": 0, "break_count": 0}

    return {
        "paragraph_count": len(root.findall(".//p")),
        "sentence_count": len(root.findall(".//s")),
        "break_count": len(root.findall(".//break")),
    }


# ---------------------------------------------------------------------------
# Timestamp correlation helpers
# ---------------------------------------------------------------------------

def _words_in_window(words: list[dict], center_sec: float, window_sec: float) -> list[dict]:
    """Return words whose time range overlaps [center - window, center + window]."""
    lo = center_sec - window_sec
    hi = center_sec + window_sec
    return [w for w in words if w["end"] >= lo and w["start"] <= hi]


# ---------------------------------------------------------------------------
# Combining detector output with transcript
# ---------------------------------------------------------------------------

def combine_with_detections(
    combined_detections: list[dict],
    words: list[dict],
    is_video: bool,
    window_sec: float = 2.0,
    rules: list[RegexRule] | None = None,
) -> dict:
    """Combine detector output with transcript analysis.

    Parameters
    ----------
    combined_detections : list[dict]
        Output from :func:`detector.combine_results`.
    words : list[dict]
        Timed word list from :func:`transcribe.transcribe_audio`.
    is_video : bool
        True when the input was a video file.
    window_sec : float
        Half-width (in seconds) of the time window used to find words spoken
        near each video detection frame.  Ignored for images.

    Returns
    -------
    dict with keys:
        text_stats  – output of :func:`analyze_text`
        ssml_stats  – output of :func:`analyze_ssml`
        frames      – list of per-frame dicts, each containing:
                        frame, timestamp_sec, ifnude, rekognition, words_spoken,
                        matched_snippets (list of snippet strings from fired rules)
    """
    text_stats = analyze_text(words)
    ssml = words_to_ssml(words)
    ssml_stats = analyze_ssml(ssml)

    frames = []
    for group in combined_detections:
        ts = group.get("timestamp_sec")

        if is_video and ts is not None:
            nearby = _words_in_window(words, ts, window_sec)
            spoken = words_to_text(nearby)
        else:
            spoken = text_stats["text"]

        frames.append({
            "frame": group["frame"],
            "timestamp_sec": ts,
            "ifnude": group["ifnude"],
            "rekognition": group["rekognition"],
            "words_spoken": spoken,
            "matched_snippets": [],
        })

    if rules:
        snippet_lists = apply_rules(frames, rules)
        for frame, snippets in zip(frames, snippet_lists):
            frame["matched_snippets"] = snippets

    return {
        "text_stats": {
            "word_count": text_stats["word_count"],
            "unique_words": text_stats["unique_words"],
            "duration_sec": text_stats["duration_sec"],
        },
        "ssml_stats": ssml_stats,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------

def format_combined_analysis(analysis: dict, is_video: bool) -> str:
    """Render a combined analysis dict as a human-readable string.

    Parameters
    ----------
    analysis : dict
        Output of :func:`combine_with_detections`.
    is_video : bool
        True when the input was a video file (affects frame headers).
    """
    lines: list[str] = []

    ts = analysis["text_stats"]
    ss = analysis["ssml_stats"]

    lines.append("=== TRANSCRIPT ANALYSIS ===")
    lines.append(f"  Words       : {ts['word_count']}  (unique: {ts['unique_words']})")
    if ts.get("duration_sec"):
        lines.append(f"  Duration    : {ts['duration_sec']}s")
    lines.append(f"  Paragraphs  : {ss['paragraph_count']}")
    lines.append(f"  Sentences   : {ss['sentence_count']}")
    lines.append(f"  Pauses      : {ss['break_count']}")

    lines.append("")
    lines.append("=== DETECTIONS WITH TRANSCRIPT CONTEXT ===")

    if not analysis["frames"]:
        lines.append("  No detection data available.")
        return "\n".join(lines)

    for group in analysis["frames"]:
        if is_video:
            lines.append(f"\n--- Frame {group['frame']} ({group['timestamp_sec']}s) ---")

        ifnude = group["ifnude"]
        rekognition = group["rekognition"]

        if not ifnude and not rekognition:
            lines.append("  No detections.")
        else:
            if ifnude:
                lines.append(f"  ifnude ({len(ifnude)}):")
                for d in ifnude:
                    box = d.get("box", [])
                    loc = (
                        ""
                        if is_video or not box
                        else f"  box=[x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}]"
                    )
                    lines.append(f"    {d['class']:<35} {d['score']:.1%}{loc}")

            if rekognition:
                lines.append(f"  Rekognition ({len(rekognition)}):")
                for d in rekognition:
                    parents = f"  <- {', '.join(d['parents'])}" if d.get("parents") else ""
                    lines.append(f"    {d['class']:<35} {d['score']:.1%}{parents}")

        spoken = group.get("words_spoken", "")
        if spoken:
            lines.append(f"  Spoken      : \"{spoken}\"")

        for snippet in group.get("matched_snippets", []):
            lines.append(f"  >> {snippet}")

    return "\n".join(lines)
