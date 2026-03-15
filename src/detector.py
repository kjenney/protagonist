#!/usr/bin/env python3
"""
Detect inappropriate body parts and general scene elements in images and videos.

Install: pip install ifnude opencv-python boto3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
import cv2
from ifnude.detector import detect as _ifnude_detect, censor as _ifnude_censor

# ---------------------------------------------------------------------------
# Logging setup – writes to debug.log
# ---------------------------------------------------------------------------

_LOG_PATH = Path(__file__).parent / "debug.log"
_log = logging.getLogger(__name__)
if not _log.handlers:
    _fh = logging.FileHandler(_LOG_PATH)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)
    # Set DEBUG=1 in the environment to re-enable debug logging
    _log.setLevel(logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.WARNING)

_log.debug("detector module loaded, logging to %s", _LOG_PATH)

# Classes returned by ifnude (EXPOSED_BELLY is filtered internally by ifnude)
EXPLICIT_CLASSES = {
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_BREAST_F",
    "EXPOSED_BREAST_M",
}

ALL_CLASSES = {
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_BREAST_F",
    "EXPOSED_BREAST_M",
    "EXPOSED_BUTTOCKS",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

IFNUDE_MODES = ["default", "fast"]


# ---------------------------------------------------------------------------
# ifnude detection
# ---------------------------------------------------------------------------

def ifnude_detect(image_path: str, min_confidence: float, classes: set, mode: str = "default") -> list[dict]:
    return [
        {"class": r["label"], "score": r["score"], "box": r["box"]}
        for r in _ifnude_detect(str(image_path), mode=mode, min_prob=min_confidence)
        if r["label"] in classes
    ]


def ifnude_detect_video(
    video_path: str,
    min_confidence: float,
    classes: set,
    frame_interval: int = 30,
    mode: str = "default",
) -> list[dict]:
    _log.debug(
        "ifnude_detect_video start: path=%s min_confidence=%s classes=%s frame_interval=%s mode=%s",
        video_path, min_confidence, classes, frame_interval, mode,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _log.debug("ifnude_detect_video: fps=%.2f total_frames=%d", fps, total_frames)
    all_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_interval == 0:
                _log.debug("ifnude: analysing frame %d (%.2fs)", frame_number, frame_number / fps)
                raw = _ifnude_detect(frame, mode=mode, min_prob=min_confidence)
                _log.debug("ifnude: frame %d raw results=%s", frame_number, raw)
                for r in raw:
                    if r["label"] in classes:
                        det = {
                            "class": r["label"],
                            "score": r["score"],
                            "box": r["box"],
                            "frame": frame_number,
                            "timestamp_sec": round(frame_number / fps, 2),
                        }
                        _log.debug("ifnude: detection accepted: %s", det)
                        all_detections.append(det)
            frame_number += 1
    finally:
        cap.release()

    _log.debug("ifnude_detect_video done: %d detections across %d frames", len(all_detections), frame_number)
    return all_detections


def censor(image_path: str, output_path: str, classes: set, mode: str = "default") -> None:
    result = _ifnude_censor(str(image_path), parts_to_blur=list(classes))
    if result is not None:
        cv2.imwrite(output_path, result)


# ---------------------------------------------------------------------------
# Rekognition detection
# ---------------------------------------------------------------------------

def _rekognition_client(profile: str | None):
    _log.debug("rekognition: creating client with profile=%s", profile)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("rekognition")
    _log.debug("rekognition: client created, region=%s", client.meta.region_name)
    return client


def _parse_rekognition_labels(response: dict, min_confidence: float) -> list[dict]:
    results = []
    for label in response.get("Labels", []):
        if label["Confidence"] / 100 >= min_confidence:
            results.append({
                "source": "rekognition",
                "class": label["Name"],
                "score": round(label["Confidence"] / 100, 4),
                "parents": [p["Name"] for p in label.get("Parents", [])],
                "categories": [c["Name"] for c in label.get("Categories", [])],
            })
    return results


def rekognition_detect(
    image_path: str,
    min_confidence: float,
    profile: str | None,
    max_labels: int = 50,
) -> list[dict]:
    client = _rekognition_client(profile)
    with open(image_path, "rb") as f:
        response = client.detect_labels(
            Image={"Bytes": f.read()},
            MaxLabels=max_labels,
            MinConfidence=min_confidence * 100,
        )
    return _parse_rekognition_labels(response, min_confidence)


def rekognition_detect_video(
    video_path: str,
    min_confidence: float,
    profile: str | None,
    frame_interval: int = 30,
    max_labels: int = 50,
) -> list[dict]:
    _log.debug(
        "rekognition_detect_video start: path=%s min_confidence=%s profile=%s frame_interval=%s max_labels=%s",
        video_path, min_confidence, profile, frame_interval, max_labels,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _log.debug("rekognition_detect_video: fps=%.2f total_frames=%d", fps, total_frames)
    client = _rekognition_client(profile)
    all_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_interval == 0:
                _log.debug("rekognition: calling detect_labels for frame %d (%.2fs)", frame_number, frame_number / fps)
                _, encoded = cv2.imencode(".jpg", frame)
                response = client.detect_labels(
                    Image={"Bytes": encoded.tobytes()},
                    MaxLabels=max_labels,
                    MinConfidence=min_confidence * 100,
                )
                _log.debug(
                    "rekognition: frame %d response: LabelModelVersion=%s Labels=%s",
                    frame_number,
                    response.get("LabelModelVersion"),
                    [{"Name": l["Name"], "Confidence": l["Confidence"]} for l in response.get("Labels", [])],
                )
                parsed = _parse_rekognition_labels(response, min_confidence)
                for r in parsed:
                    r["frame"] = frame_number
                    r["timestamp_sec"] = round(frame_number / fps, 2)
                    _log.debug("rekognition: detection accepted: %s", r)
                    all_detections.append(r)
            frame_number += 1
    finally:
        cap.release()

    _log.debug("rekognition_detect_video done: %d detections across %d frames", len(all_detections), frame_number)
    return all_detections


# ---------------------------------------------------------------------------
# Combining and output
# ---------------------------------------------------------------------------

def combine_results(ifnude: list[dict], rekognition: list[dict], is_video: bool) -> list[dict]:
    """Group ifnude and Rekognition detections by frame (video) or into a single group (image)."""
    if not is_video:
        return [{"frame": None, "timestamp_sec": None, "ifnude": ifnude, "rekognition": rekognition}]

    frames: dict[int, dict] = {}
    for d in ifnude:
        f = d["frame"]
        if f not in frames:
            frames[f] = {"frame": f, "timestamp_sec": d["timestamp_sec"], "ifnude": [], "rekognition": []}
        frames[f]["ifnude"].append(d)
    for d in rekognition:
        f = d["frame"]
        if f not in frames:
            frames[f] = {"frame": f, "timestamp_sec": d["timestamp_sec"], "ifnude": [], "rekognition": []}
        frames[f]["rekognition"].append(d)

    return [frames[f] for f in sorted(frames)]


def print_combined_results(combined: list[dict], is_video: bool) -> None:
    for group in combined:
        if is_video:
            print(f"\n--- Frame {group['frame']} ({group['timestamp_sec']}s) ---")
        else:
            print()

        ifnude = group["ifnude"]
        rekognition = group["rekognition"]

        if not ifnude and not rekognition:
            print("  No detections.")
            continue

        if ifnude:
            print(f"  ifnude ({len(ifnude)}):")
            for d in ifnude:
                box = d["box"]
                loc = "" if is_video else f"  box=[x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}]"
                print(f"    {d['class']:<35} {d['score']:.1%}{loc}")

        if rekognition:
            print(f"  Rekognition ({len(rekognition)}):")
            for d in rekognition:
                parents = f"  <- {', '.join(d['parents'])}" if d["parents"] else ""
                print(f"    {d['class']:<35} {d['score']:.1%}{parents}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect body parts (NudeNet) and scene elements (Rekognition) in images/videos"
    )
    parser.add_argument("input", help="Path to an image or video file")
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum confidence threshold for both detectors (default: 0.5)",
    )
    parser.add_argument(
        "--all-classes", action="store_true",
        help="NudeNet: detect all body part classes, not just explicit ones",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=30,
        help="Video: analyse every Nth frame (default: 30)",
    )
    parser.add_argument(
        "--censor", metavar="OUTPUT_PATH",
        help="Write a censored image to this path (images only, NudeNet)",
    )
    parser.add_argument(
        "--profile", help="AWS profile name for Rekognition",
    )
    parser.add_argument(
        "--no-nudenet", action="store_true", help="Skip NudeNet detection",
    )
    parser.add_argument(
        "--no-rekognition", action="store_true", help="Skip Rekognition detection",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    classes = ALL_CLASSES if args.all_classes else EXPLICIT_CLASSES
    is_video = Path(args.input).suffix.lower() in VIDEO_EXTENSIONS

    nudenet_results = []
    rekognition_results = []

    try:
        if not args.no_nudenet:
            if is_video:
                nudenet_results = nudenet_detect_video(
                    args.input, args.min_confidence, classes, args.frame_interval
                )
            else:
                nudenet_results = nudenet_detect(args.input, args.min_confidence, classes)

        if not args.no_rekognition:
            if is_video:
                rekognition_results = rekognition_detect_video(
                    args.input, args.min_confidence, args.profile, args.frame_interval
                )
            else:
                rekognition_results = rekognition_detect(
                    args.input, args.min_confidence, args.profile
                )

        combined = combine_results(nudenet_results, rekognition_results, is_video)

        if args.json:
            print(json.dumps(combined, indent=2))
        else:
            print_combined_results(combined, is_video)

        if args.censor and not is_video:
            censor(args.input, args.censor, classes)
            print(f"\nCensored image saved to: {args.censor}")

        # Exit 1 if explicit content detected
        sys.exit(1 if nudenet_results else 0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
