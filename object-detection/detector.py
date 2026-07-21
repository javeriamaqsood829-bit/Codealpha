"""
detector.py
Loads a pre-trained YOLOv4-tiny model (via OpenCV's DNN module) and runs
object detection on individual frames. No PyTorch/TensorFlow required —
lightweight and fast enough to run on free-tier cloud hosting.
"""

import cv2
import numpy as np
import os

MODEL_DIR = "model"
CONFIG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)

_net = None
_output_layers = None
_classes = None


def _load():
    global _net, _output_layers, _classes
    if _net is None:
        _net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = _net.getLayerNames()
        out_indices = _net.getUnconnectedOutLayers()
        _output_layers = [layer_names[i - 1] for i in out_indices]

        with open(NAMES_PATH, "r") as f:
            _classes = [line.strip() for line in f.readlines()]

    return _net, _output_layers, _classes


def get_classes():
    _, _, classes = _load()
    return classes


def detect(frame, conf_threshold=CONF_THRESHOLD, classes_filter=None):
    """
    Run YOLO detection on a single BGR frame (numpy array).
    Returns a list of detections: [(x, y, w, h), confidence, class_name]
    classes_filter: optional set of class names to keep (None = keep all)
    """
    net, output_layers, classes = _load()
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = classes[class_ids[i]]
            if classes_filter and class_name not in classes_filter:
                continue
            results.append({
                "box": boxes[i],
                "confidence": confidences[i],
                "class_name": class_name
            })

    return results
