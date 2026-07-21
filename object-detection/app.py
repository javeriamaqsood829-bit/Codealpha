import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

from pipeline import process_video, process_image
from detector import get_classes

st.set_page_config(
    page_title="AI Object Detector",
    page_icon="🎯",
    layout="wide"
)

# ---------------- CUSTOM CSS (Premium Dark Gradient Theme) ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { padding-top: 2rem; max-width: 1050px; }
    h1, h2, h3 { font-family: 'Sora', sans-serif !important; }

    .nav-badge {
        display: inline-block;
        font-size: 12px;
        color: #c8bfff;
        background: rgba(139,124,246,0.15);
        border: 1px solid rgba(139,124,246,0.35);
        padding: 6px 14px;
        border-radius: 999px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(135deg, #ffffff, #c8bfff 60%, #8b7cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .hero-sub {
        font-size: 16px;
        color: #b8b3d9;
        line-height: 1.6;
        margin-bottom: 10px;
        max-width: 640px;
    }
    .stats-row { display: flex; gap: 50px; margin: 26px 0 36px; flex-wrap: wrap; }
    .stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #8b7cf6, #c8bfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 12px; color: #8a84b8; margin-top: 2px; }

    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px 16px;
        height: 100%;
    }
    .feature-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, rgba(106,90,205,0.3), rgba(139,124,246,0.3));
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 17px;
        margin-bottom: 10px;
    }
    .feature-card h4 { font-size: 13px; margin: 0 0 6px; }
    .feature-card p { font-size: 12px; color: #8a84b8; line-height: 1.5; margin: 0; }

    .app-shell {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 28px;
        margin-top: 20px;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6a5acd, #8b7cf6);
        color: white; border: none; border-radius: 999px;
        padding: 10px 26px; font-weight: 700; font-size: 14px;
        box-shadow: 0 6px 24px rgba(106,90,205,0.4);
    }
    div[data-testid="stDownloadButton"] button {
        background: rgba(255,255,255,0.06);
        color: white; border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
    }
    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(139,124,246,0.4);
        border-radius: 14px;
    }
    .stat-chip {
        display: inline-block;
        background: rgba(139,124,246,0.15);
        border: 1px solid rgba(139,124,246,0.3);
        color: #c8bfff;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #6a5acd; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown('<div class="nav-badge">⚡ Powered by YOLOv4-tiny + Centroid Tracking</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🎯 AI Object Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Detect and track objects in real time — upload a photo or video and '
    'watch a neural network find, label, and follow every object with a persistent ID.</div>',
    unsafe_allow_html=True
)

all_classes = get_classes()
st.markdown(f"""
<div class="stats-row">
    <div><div class="stat-num">{len(all_classes)}</div><div class="stat-label">Object classes</div></div>
    <div><div class="stat-num">CPU</div><div class="stat-label">No GPU required</div></div>
    <div><div class="stat-num">Live</div><div class="stat-label">ID tracking</div></div>
    <div><div class="stat-num">MP4</div><div class="stat-label">Export ready</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
f1, f2, f3, f4 = st.columns(4)
features = [
    ("🧠", "YOLOv4-tiny detector", "Fast, lightweight neural network trained on 80 everyday object classes."),
    ("🔗", "Persistent tracking", "Each object keeps its own ID as it moves across frames."),
    ("🎚️", "Adjustable confidence", "Tune sensitivity to catch more objects or reduce false positives."),
    ("⬇️", "Export annotated video", "Download the fully processed, labeled video when done."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- APP SHELL ----------------
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

tab_video, tab_image = st.tabs(["🎬 Video (Detection + Tracking)", "🖼️ Image (Quick Detection)"])

with tab_video:
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        video_file = st.file_uploader("Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    with col_settings:
        confidence = st.slider("Detection confidence", 0.2, 0.9, 0.4, 0.05)

    if video_file is not None:
        if st.button("🎯 Run Detection + Tracking", key="run_video"):
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_in.write(video_file.read())
            tmp_in.close()

            tmp_out_path = tmp_in.name.replace(".mp4", "_out.mp4")

            progress_bar = st.progress(0.0, text="Processing video...")

            def update_progress(frac):
                progress_bar.progress(frac, text=f"Processing video... {int(frac*100)}%")

            with st.spinner("Analyzing frames..."):
                stats = process_video(
                    tmp_in.name, tmp_out_path,
                    conf_threshold=confidence,
                    progress_callback=update_progress
                )

            progress_bar.empty()
            st.success(f"Done! Processed {stats['total_frames']} frames, "
                       f"tracked {stats['unique_objects_seen']} unique objects.")

            # Re-encode for browser compatibility
            reencoded_path = tmp_out_path.replace("_out.mp4", "_web.mp4")
            os.system(f'ffmpeg -y -i "{tmp_out_path}" -vcodec libx264 -acodec aac "{reencoded_path}" -loglevel quiet')
            final_path = reencoded_path if os.path.exists(reencoded_path) else tmp_out_path

            st.video(final_path)

            if stats["class_counts"]:
                chips = "".join(
                    f'<span class="stat-chip">{name}: {count}</span>'
                    for name, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1])
                )
                st.markdown(chips, unsafe_allow_html=True)

            with open(final_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Annotated Video",
                    data=f.read(),
                    file_name="detected_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            os.unlink(tmp_in.name)
    else:
        st.info("👆 Upload a video to detect and track objects frame by frame.")

with tab_image:
    col_upload2, col_settings2 = st.columns([2, 1])
    with col_upload2:
        image_source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
        image_file = None
        if image_source == "Upload a photo":
            image_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="img_upload")
        else:
            image_file = st.camera_input("Take a photo")
    with col_settings2:
        confidence_img = st.slider("Detection confidence", 0.2, 0.9, 0.4, 0.05, key="conf_img")

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting objects..."):
            annotated, detections = process_image(img.copy(), conf_threshold=confidence_img)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if detections:
            st.markdown(f"### ✅ {len(detections)} object(s) detected")

            counts = {}
            for d in detections:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
            chips = "".join(
                f'<span class="stat-chip">{name}: {count}</span>'
                for name, count in sorted(counts.items(), key=lambda x: -x[1])
            )
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("**Detected & tracked objects:**")
            for d in detections:
                st.markdown(
                    f"- 🔹 **{d['class_name']} #{d['id']}** — confidence {d['confidence']:.0%}"
                )
        else:
            st.warning("No objects detected above the current confidence threshold. Try lowering it in the slider.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HOW IT WORKS ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## How it works")
h1, h2, h3 = st.columns(3)
steps = [
    ("1", "Detect", "YOLOv4-tiny scans each frame and finds objects from 80 everyday classes."),
    ("2", "Track", "A centroid-based tracker follows each object across frames, assigning a stable ID."),
    ("3", "Export", "Every detection is drawn with a label and ID, then saved to a downloadable video."),
]
for col, (num, title, desc) in zip([h1, h2, h3], steps):
    with col:
        st.markdown(f"""
        <div class="feature-card" style="text-align:left;">
            <div class="feature-icon">{num}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;color:#6b6591;font-size:12px;margin-top:40px;padding-top:16px;'
    'border-top:1px solid rgba(255,255,255,0.06);">Built with YOLOv4-tiny · OpenCV · Streamlit — '
    'CodeAlpha AI Internship, Task 4</div>',
    unsafe_allow_html=True
)
