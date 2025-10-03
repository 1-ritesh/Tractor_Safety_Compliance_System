import streamlit as st
import os
from utils import extract_best_frame

st.set_page_config(page_title="Tractor Frame Extractor", layout="centered")
st.title("🚜 Tractor Best Frame Extractor")
st.markdown("Upload a CCTV video of tractor's backside and get the best frame with number plate & red cloth clearly visible.")

uploaded_video = st.file_uploader("📁 Upload Tractor Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success("✅ Video uploaded successfully!")

    if st.button("🎯 Extract Best Frame"):
        with st.spinner("Processing video..."):
            best_frame_path = extract_best_frame(video_path)
            if best_frame_path:
                st.success("✅ Best frame extracted successfully!")
                st.image(best_frame_path, caption="Best Frame Detected", use_column_width=True)
            else:
                st.error("❌ No tractor detected in the video.")
