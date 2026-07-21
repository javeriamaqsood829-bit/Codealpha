# 🎯 AI Object Detector

A CodeAlpha AI Internship project — **Task 4: Object Detection and Tracking**.

Detects and tracks objects in videos or photos using **YOLOv4-tiny**
(via OpenCV's DNN module — no PyTorch/TensorFlow required) combined
with a lightweight **centroid-based multi-object tracker** that assigns
each detected object a persistent ID as it moves across frames.

## How it works

1. **`detector.py`** — loads YOLOv4-tiny and runs detection on a single
   frame, returning bounding boxes, confidence scores, and class labels
   (80 everyday object classes from the COCO dataset).
2. **`tracker.py`** — a centroid-based tracker (inspired by SORT) that
   matches detections across frames by proximity, giving each object a
   stable ID for as long as it stays in view.
3. **`pipeline.py`** — combines detection + tracking frame-by-frame for
   a full video, draws labeled boxes with IDs, and writes an annotated
   output video. Also supports single-image detection.
4. **`app.py`** — a Streamlit web interface with two modes:
   - **Video** — upload a clip, get back an annotated video with
     tracked object IDs + a summary of detected object counts.
   - **Image** — upload a photo or use your camera for instant
     detection (no tracking needed for a single frame).

## Features
- YOLOv4-tiny detector (80 COCO object classes)
- Persistent object tracking with IDs across video frames
- Adjustable detection confidence threshold
- Object count summary per video/photo
- Downloadable annotated output video
- Live camera snapshot detection
- Runs entirely on CPU — no GPU needed

## Run locally

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

> Note: video re-encoding for browser playback uses `ffmpeg`. If it's
> not installed locally, the app will still work but browser preview
> of the output video may not play (the download will still be valid).
> Install ffmpeg from https://ffmpeg.org/download.html if needed.

## Deployment
Deployed free on [Streamlit Community Cloud](https://share.streamlit.io).
`packages.txt` tells Streamlit Cloud to install `ffmpeg` automatically.

## Tech Stack
- OpenCV DNN (YOLOv4-tiny)
- SciPy (centroid distance matching for tracking)
- Streamlit (web interface)
- FFmpeg (video re-encoding for browser playback)

---
Made with ❤️ by Javeria — CodeAlpha AI Internship, Task 4
