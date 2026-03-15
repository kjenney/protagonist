#!/usr/bin/env python3
"""
User-defined regex rules that fire custom snippets when both a transcript
pattern and a label pattern match a detection frame.

A rule fires for a frame when:
  - transcript_pattern is empty OR it matches the frame's words_spoken text
  - label_pattern is empty OR it matches the concatenated label class names
    (from both NudeNet and Rekognition detections) for that frame

Both conditions must be satisfied (AND logic).  An empty pattern string is
treated as "always matches", so a rule with only a transcript_pattern is
effectively a pure transcript search, and vice-versa.

Public API
----------
RegexRule
    Dataclass representing a single user rule.

apply_rules(frames, rules)
    Apply a list of RegexRule objects to a list of combined-detection frames.
    Returns a list of matched-snippet lists (one entry per frame).
"""

import re
from dataclasses import dataclass, field


@dataclass
class RegexRule:
    """A user-defined rule that fires a snippet when both patterns match.

    Attributes
    ----------
    name : str
        Human-readable label for this rule (shown in output).
    transcript_pattern : str
        Regular expression tested against the ``words_spoken`` field of each
        frame (case-insensitive).  Empty string → always matches.
    label_pattern : str
        Regular expression tested against a space-joined string of all
        detection class names in the frame (case-insensitive).
        Empty string → always matches.
    snippet : str
        Text to inject into the output when the rule fires.
    """

    name: str
    transcript_pattern: str
    label_pattern: str
    snippet: str


def apply_rules(frames: list[dict], rules: list[RegexRule]) -> list[list[str]]:
    """Apply regex rules to each frame and return matched snippets.

    Parameters
    ----------
    frames : list[dict]
        Per-frame dicts from :func:`text_analysis.combine_with_detections`,
        each containing ``words_spoken``, ``ifnude``, and ``rekognition``.
    rules : list[RegexRule]
        Rules to evaluate against each frame.

    Returns
    -------
    list[list[str]]
        One inner list per frame.  Each inner list contains the ``snippet``
        strings of every rule that fired for that frame (may be empty).
    """
    results: list[list[str]] = []

    for frame in frames:
        spoken = frame.get("words_spoken", "")
        label_classes = [d["class"] for d in frame.get("ifnude", [])] + [
            d["class"] for d in frame.get("rekognition", [])
        ]
        labels_text = " ".join(label_classes)

        matched: list[str] = []
        for rule in rules:
            transcript_ok = (
                not rule.transcript_pattern
                or bool(re.search(rule.transcript_pattern, spoken, re.IGNORECASE))
            )
            label_ok = (
                not rule.label_pattern
                or bool(re.search(rule.label_pattern, labels_text, re.IGNORECASE))
            )
            if transcript_ok and label_ok:
                matched.append(rule.snippet)

        results.append(matched)

    return results
