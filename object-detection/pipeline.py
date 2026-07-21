"""
pipeline.py
Combines the YOLO detector + centroid tracker to process a full video:
reads frames, detects objects, tracks them across frames with persistent
IDs, draws bounding boxes + labels, and writes an annotated output video.
"""

import cv2
import numpy as np
import random

from detector import detect, get_classes
from tracker import CentroidTracker

random.seed(7)
_CLASS_COLORS = {}


def _color_for(label):
    if label not in _CLASS_COLORS:
        _CLASS_COLORS[label] = tuple(random.randint(80, 255) for _ in range(3))
    return _CLASS_COLORS[label]


def draw_tracked_objects(frame, tracked, show_labels=True):
    for object_id, info in tracked.items():
        x, y, w, h = info["box"]
        label = info["label"]
        color = _color_for(label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if show_labels:
            text = f"{label} #{object_id}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
            cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def process_video(input_path, output_path, conf_threshold=0.4,
                   classes_filter=None, progress_callback=None,
                   max_disappeared=15, max_distance=80, frame_skip=1):
    """
    Processes a video file end-to-end: detection + tracking + annotation.
    progress_callback(fraction_done) is called periodically if provided.
    Returns: dict with stats (total_frames, unique_objects_seen, class_counts)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
    seen_ids = set()
    class_counts = {}

    frame_idx = 0
    last_tracked = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            detections = detect(frame, conf_threshold=conf_threshold, classes_filter=classes_filter)
            tracked = tracker.update(detections)
            last_tracked = tracked
        else:
            tracked = last_tracked

        for object_id, info in tracked.items():
            if object_id not in seen_ids:
                seen_ids.add(object_id)
                class_counts[info["label"]] = class_counts.get(info["label"], 0) + 1

        frame = draw_tracked_objects(frame, tracked)
        writer.write(frame)

        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()

    return {
        "total_frames": frame_idx,
        "unique_objects_seen": len(seen_ids),
        "class_counts": class_counts
    }


def process_image(image, conf_threshold=0.4, classes_filter=None):
    """
    Runs detection on a single image and draws each detected object with
    its own sequential ID (e.g. "person #1", "car #2") so the output
    reads the same way as the video tracker's labels. Returns the
    annotated image + list of detections (each with an assigned id).
    """
    detections = detect(image, conf_threshold=conf_threshold, classes_filter=classes_filter)

    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["box"]
        label = det["class_name"]
        conf = det["confidence"]
        det["id"] = idx
        color = _color_for(label)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label} #{idx} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(image, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)

    return image, detections
