# app_callbacks.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import time
import numpy as np
import json
from vosk import Model, KaldiRecognizer
import mediapipe as mp
import threading

# --- Emotion utilities (same heuristics) ---
mp_face = mp.solutions.face_mesh
_face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return coords

def mouth_aspect_ratio(lm):
    left, right = lm[61], lm[291]
    top, bottom = lm[13], lm[14]
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - bottom)
    return 0.0 if width == 0 else height / width

def eye_aspect_ratio(lm, left=True):
    if left:
        upper, lower = lm[159], lm[145]
        corner1, corner2 = lm[33], lm[133]
    else:
        upper, lower = lm[386], lm[374]
        corner1, corner2 = lm[263], lm[362]
    vert = np.linalg.norm(upper - lower)
    hor = np.linalg.norm(corner2 - corner1)
    return 0.0 if hor == 0 else vert / hor

def eyebrow_distance(lm):
    return np.linalg.norm(lm[70] - lm[300])

def mouth_corner_slope(lm):
    left, right = lm[61], lm[291]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return 0.0 if dx == 0 else dy / dx

def detect_emotion_from_frame(frame,
                              thresh_smile=0.035,
                              thresh_surprise=0.25,
                              thresh_anger=0.05,
                              thresh_sad_slope=0.02) -> str | None:
    lm = get_landmarks(frame)
    if lm is None:
        return None
    if mouth_aspect_ratio(lm) > thresh_smile:
        return "happy"
    ear = (eye_aspect_ratio(lm, True) + eye_aspect_ratio(lm, False)) / 2.0
    if ear > thresh_surprise:
        return "surprised"
    face_width = np.linalg.norm(lm[234] - lm[454])
    if face_width > 0 and eyebrow_distance(lm) < thresh_anger * face_width:
        return "angry"
    if mouth_corner_slope(lm) > thresh_sad_slope:
        return "sad"
    return "neutral"

# --- Vosk model load ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
@st.cache_resource
def load_vosk_model(path):
    try:
        return Model(path)
    except Exception as e:
        st.error(f"Failed loading Vosk model at '{path}': {e}")
        return None

# --- Shared buffers preserved across reruns ---
@st.cache_resource
def get_shared_buffers():
    return {
        "audio": [],           # list of numpy arrays (PCM)
        "emotion": [],         # list of emotion strings
        "last_emotion_ts": time.time(),
        "lock": threading.Lock()
    }

buffers = get_shared_buffers()

# --- Audio callback: buffer raw audio frames ---
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Called in worker thread for each audio frame.
    We append PCM data to buffers["audio"] under lock.
    No Streamlit state writes here.
    """
    pcm = frame.to_ndarray()  # shape (n_samples, n_channels)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    with buffers["lock"]:
        buffers["audio"].append(pcm.copy())
    return frame

# --- Video callback: sample emotion every 1s, overlay timestamp+emotion ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Called in worker thread for each video frame.
    Every ~1s, detect emotion and append to buffers["emotion"] under lock.
    Overlay timestamp and last emotion on the frame.
    """
    img = frame.to_ndarray(format="bgr24")
    now = time.time()
    # sample every 1 second
    with buffers["lock"]:
        last_ts = buffers["last_emotion_ts"]
    if now - last_ts >= 1.0:
        emotion = detect_emotion_from_frame(img)
        with buffers["lock"]:
            if emotion is not None:
                # append only if different from last or if empty
                if not buffers["emotion"] or emotion != buffers["emotion"][-1]:
                    buffers["emotion"].append(emotion)
            buffers["last_emotion_ts"] = now
    # overlay timestamp and last emotion
    with buffers["lock"]:
        label = buffers["emotion"][-1] if buffers["emotion"] else "No face"
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, label, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Transcribe function (on Fetch) ---
def transcribe_buffered_audio():
    """
    Combine all buffered PCM chunks into one array,
    feed to Vosk in one pass, return transcript string.
    Clears buffers["audio"].
    """
    model = load_vosk_model(VOSK_MODEL_PATH)
    if model is None:
        return ""
    with buffers["lock"]:
        if not buffers["audio"]:
            return ""
        # concatenate all chunks
        pcm_list = buffers["audio"]
        buffers["audio"] = []
    # Vosk expects 16kHz; ensure frames are 16kHz from WebRTC settings
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)
    # Feed concatenated PCM bytes
    for chunk in pcm_list:
        data = chunk.tobytes()
        recognizer.AcceptWaveform(data)
    res = recognizer.FinalResult()
    try:
        text = json.loads(res).get("text", "")
    except Exception:
        text = ""
    return text

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Edge AI Callbacks")
st.title("🎥 + 🎤 → Text & 😊 Emotion (Callbacks)")

# Session state for stored results
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "emotions" not in st.session_state:
    st.session_state.emotions = []

col_vid, col_info = st.columns([2, 1])

with col_info:
    st.markdown("## Controls")
    if st.button("Clear Stored Transcript & Emotions"):
        st.session_state.transcript = ""
        st.session_state.emotions = []

    st.markdown("## Stored Results")
    transcript_box = st.empty()
    emotion_box = st.empty()

with col_vid:
    st.markdown("## Webcam + Mic Stream")
    # Always mount webrtc_streamer with fixed key & callbacks
    webrtc_ctx = webrtc_streamer(
        key="callback-edge-ai",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
        video_html_attrs={"style": "width: 100%; height: auto;"},
    )

# Fetch button to pull from shared buffers
with col_info:
    if st.button("Fetch Transcript & Emotions"):
        # Transcribe audio buffer
        new_text = transcribe_buffered_audio()
        if new_text:
            st.session_state.transcript = (
                st.session_state.transcript + " " + new_text
            ).strip()
        # Pull emotion buffer
        with buffers["lock"]:
            new_emots = buffers["emotion"][:]
            # Do not clear emotion buffer entirely; keep history,
            # but we store a copy in session_state.
        # Update session_state.emotions to match buffers["emotion"]
        st.session_state.emotions = new_emots.copy()

# Display stored
with col_info:
    transcript_box.text_area("Transcript", value=st.session_state.transcript, height=200)
    if st.session_state.emotions:
        emotion_box.write(", ".join(st.session_state.emotions[-50:]))
    else:emotion_box.write("No emotions yet.")
