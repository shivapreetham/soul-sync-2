# webrtc_test.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time

st.title("📷 Webcam Stream (WebRTC)")

# Setup: init session state
if "start_cam" not in st.session_state:
    st.session_state["start_cam"] = False

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI buttons
col1, col2 = st.columns(2)
if col1.button("▶️ Start Webcam"):
    st.session_state["start_cam"] = True

if col2.button("⏹️ Stop Webcam"):
    st.session_state["start_cam"] = False

# Display webcam only if enabled
if st.session_state["start_cam"]:
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.info("Webcam is off. Click ▶️ to start.")
