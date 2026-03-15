#!/usr/bin/env python3
"""
AWS Polly text-to-speech synthesis for subtitle entries.

Public API
----------
synthesize_subtitles(entries, voice_id, profile, region)
    Synthesize subtitle text entries to speech and return the MP3 file path.
"""

import io
import tempfile
from typing import Optional


# A representative selection of Polly neural/standard voices.
POLLY_VOICES = [
    "Amy", "Aria", "Astrid", "Ayanda", "Brian", "Camila", "Carla",
    "Celine", "Conchita", "Emma", "Enrique", "Ewa", "Filiz",
    "Gabrielle", "Giorgio", "Hans", "Ines", "Ivy", "Jacek", "Jan",
    "Joanna", "Joey", "Justin", "Karl", "Kendra", "Kevin", "Kimberly",
    "Lea", "Liam", "Liv", "Lotte", "Lucia", "Lupe", "Mads", "Maja",
    "Marlene", "Mathieu", "Matthew", "Maxim", "Mia", "Miguel", "Mizuki",
    "Naja", "Nicole", "Olivia", "Penelope", "Raveena", "Ricardo",
    "Ruben", "Russell", "Ruth", "Salli", "Seoyeon", "Stephen", "Takumi",
    "Tatyana", "Vicki", "Vitoria", "Zeina", "Zhiyu",
]


def _polly_client(profile: Optional[str] = None, region: str = "us-east-1"):
    """Create a boto3 Polly client with optional profile and region."""
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("polly")


def _escape_ssml(text: str) -> str:
    """Escape characters that are special in SSML/XML."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def synthesize_subtitles(
    entries: list[dict],
    voice_id: str = "Joanna",
    profile: Optional[str] = None,
    region: str = "us-east-1",
) -> str:
    """Synthesize subtitle text entries to speech using AWS Polly.

    Each entry is synthesized individually.  Silence is inserted before
    each entry so that its speech starts at the entry's ``start`` timestamp,
    making the audio track align with the video when played simultaneously.

    Parameters
    ----------
    entries : list[dict]
        Subtitle entries produced by
        :func:`create_subtitles.frames_to_subtitle_entries`.  Each entry
        must have a ``text`` key and a ``start`` key (seconds as float).
    voice_id : str
        AWS Polly voice ID (e.g. ``"Joanna"``, ``"Matthew"``).  Must be a
        voice that supports the SSML text type.
    profile : str or None
        AWS profile name to use, or ``None`` for the default credential chain.
    region : str
        AWS region for the Polly endpoint (default ``"us-east-1"``).

    Returns
    -------
    str
        Absolute path to a temporary MP3 file containing the synthesized
        speech with timing-aligned silence.

    Raises
    ------
    ValueError
        If *entries* is empty or none of the entries contain usable text.
    """
    from pydub import AudioSegment

    usable = [e for e in entries if e.get("text", "").strip()]
    if not usable:
        raise ValueError("No subtitle text found to synthesize.")

    client = _polly_client(profile=profile, region=region)
    result = AudioSegment.empty()
    cursor_ms = 0  # current end position of built-up audio, in milliseconds

    for entry in usable:
        start_ms = int(float(entry["start"]) * 1000)

        # Pad with silence so this segment starts at its subtitle timestamp.
        silence_ms = max(0, start_ms - cursor_ms)
        if silence_ms > 0:
            result += AudioSegment.silent(duration=silence_ms)

        # Synthesize this entry's speech.
        ssml = f"<speak><p>{_escape_ssml(entry['text'].strip())}</p></speak>"
        response = client.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId=voice_id,
        )
        segment = AudioSegment.from_mp3(io.BytesIO(response["AudioStream"].read()))
        result += segment
        cursor_ms = start_ms + len(segment)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    result.export(tmp.name, format="mp3")
    return tmp.name
