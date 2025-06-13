# app_asr.py

import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import json
from vosk import Model, KaldiRecognizer

# Path to Vosk model folder (download and extract it yourself)
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

@st.cache_resource
def load_vosk_model(path):
    try:
        model = Model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load Vosk model at '{path}': {e}")
        return None

def transcribe_audio(audio_data: np.ndarray, fs=16000, model=None):
    if audio_data is None or model is None:
        return ""
    rec = KaldiRecognizer(model, fs)
    rec.SetWords(True)
    data_bytes = audio_data.tobytes()
    if rec.AcceptWaveform(data_bytes):
        res = rec.Result()
    else:
        res = rec.FinalResult()
    try:
        return json.loads(res).get("text", "")
    except Exception:
        return ""

# --- SESSION STATE INIT ---
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = None

# --- UI ---
st.set_page_config(page_title="Offline ASR", page_icon="🎤")
st.title("🎤 Offline ASR with Vosk")
st.write("Click Start to begin recording. Stop to transcribe using offline speech recognition.")

# Load model
model = load_vosk_model(VOSK_MODEL_PATH)
if model is None:
    st.stop()

# --- BUTTONS ---
col1, col2 = st.columns(2)
with col1:
    st.button("🎙️ Start Recording", on_click=lambda: st.session_state.update(recording=True, audio_buffer=None))
with col2:
    st.button("🛑 Stop Recording", on_click=lambda: st.session_state.update(recording=False))

# --- RECORD ---
fs = 16000  # sample rate
if st.session_state.recording:
    st.warning("Recording... Speak now.")
    try:
        chunk_duration = 1  # seconds
        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        chunk = chunk.flatten()
        if st.session_state.audio_buffer is None:
            st.session_state.audio_buffer = chunk
        else:
            st.session_state.audio_buffer = np.concatenate([st.session_state.audio_buffer, chunk])
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Recording error: {e}")

# --- TRANSCRIBE ---
if not st.session_state.recording and st.session_state.audio_buffer is not None:
    audio_data = st.session_state.audio_buffer
    st.success("Recording stopped.")
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmpfile.name, audio_data, fs)
    st.audio(tmpfile.name, format='audio/wav')

    st.write("Transcribing...")
    text = transcribe_audio(audio_data, fs=fs, model=model)
    if text:
        st.success(f"📝 Transcription: {text}")
    else:
        st.warning("No speech recognized.")

    # Clear after transcribing
    st.session_state.audio_buffer = None
