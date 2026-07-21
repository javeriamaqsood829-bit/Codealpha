"""
tracker.py
A lightweight centroid-based multi-object tracker (inspired by SORT).
Assigns a persistent ID to each detected object and follows it across
frames by matching centroids frame-to-frame, without needing a heavy
deep-learning re-identification model (keeps things fast & deployable
on free-tier hosting).
"""

import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()       # id -> centroid (x, y)
        self.boxes = OrderedDict()         # id -> (x, y, w, h)
        self.labels = OrderedDict()        # id -> class_name
        self.disappeared = OrderedDict()   # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box, label):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.labels[self.next_object_id] = label
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.labels[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        detections: list of dicts with 'box' (x, y, w, h), 'class_name'
        Returns: dict of {object_id: {"box":..., "label":...}}
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._current_state()

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, det in enumerate(detections):
            x, y, w, h = det["box"]
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self.register(input_centroids[i], det["box"], det["class_name"])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = detections[col]["box"]
                self.labels[object_id] = detections[col]["class_name"]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], detections[col]["box"], detections[col]["class_name"])

        return self._current_state()

    def _current_state(self):
        return {
            object_id: {"box": self.boxes[object_id], "label": self.labels[object_id]}
            for object_id in self.objects
        }
