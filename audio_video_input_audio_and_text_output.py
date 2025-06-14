import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
from vosk import Model, KaldiRecognizer
import soundfile as sf
import requests
from collections import Counter
import json
import time
import threading
import queue
import os
import pyaudio
import pyttsx3
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
import logging

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Vosk model
VOSK_MODEL_PATH = "D:/College_Life/projects/Hackathon/Edge AI/soul-sync-2/vosk-model-small-en-us-0.15"
if not os.path.isdir(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}. Download from https://alphacephei.com/vosk/models")
vosk_model = Model(VOSK_MODEL_PATH)

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Global variables
recording = False
snapshots = []
audio_path = None
audio_thread = None
audio_queue = queue.Queue()
cap = None
executor = ThreadPoolExecutor(max_workers=4)  # Optimize for Snapdragon X Elite

# Custom CSS for highlighting textboxes
custom_css = """
.highlighted {
    border: 3px solid black !important;
    transition: border 0.3s ease;
}
"""

def get_landmarks(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        print("No face landmarks detected in frame")
        return None
    lm = results.multi_face_landmarks[0].landmark
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

def detect_emotion_from_frame(frame, thresh_smile=0.035, thresh_surprise=0.25, thresh_anger=0.05, thresh_sad_slope=0.02):
    try:
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
    except Exception as e:
        print(f"Emotion detection error: {str(e)}")
        return None

def record_audio():
    global audio_path
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    print("Recording audio...")
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = b''.join(frames)
    audio_path = "temp_audio.wav"
    sf.write(audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
    audio_queue.put(audio_path)
    print(f"Audio saved at: {audio_path}")

def capture_snapshots():
    global snapshots, cap
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return
    last_time = time.time()
    print("Capturing snapshots...")
    while recording and len(snapshots) < 10:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        current_time = time.time()
        if current_time - last_time >= 1.0:
            snapshots.append(frame.copy())
            last_time = current_time

def video_stream():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
        time.sleep(0.033)  # ~30 FPS

def start_recording():
    global recording, snapshots, audio_thread
    if recording:
        print("Already recording")
        return "Already recording"
    recording = True
    snapshots = []
    print("Starting snapshot capture thread...")
    threading.Thread(target=capture_snapshots, daemon=True).start()
    print("Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()
    return "Recording audio and video... Speak clearly for 10–15 seconds."

async def process_emotions(snapshots):
    print("Processing emotions...")
    start = time.time()
    emotions = list(executor.map(detect_emotion_from_frame, snapshots))
    emotions = [e for e in emotions if e is not None]
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "No face detected"
    print(f"Detected emotions: {emotions}, Most common: {most_common_emotion}, Time: {time.time() - start:.2f} seconds")
    return most_common_emotion

async def transcribe_audio(audio_path):
    print("Transcribing audio...")
    start = time.time()
    try:
        data, samplerate = sf.read(audio_path)
        print(f"Audio shape: {data.shape}, Sample rate: {samplerate}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = (data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(vosk_model, samplerate)
        if not recognizer.AcceptWaveform(data.tobytes()):
            print("Vosk failed to process audio")
            return "No speech detected"
        result = json.loads(recognizer.Result())
        text = result.get("text", "") or "No speech detected"
        print(f"Transcription: {text}, Time: {time.time() - start:.2f} seconds")
        return text
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return f"Transcription error: {str(e)}"

async def send_to_gemini(text):
    print("Sending to Gemini server...")
    start = time.time()
    payload = {"message": text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://sign-language-3-5vax.onrender.com/gemini", json=payload, timeout=3) as response:
                response.raise_for_status()
                data = await response.json()
                response_text = data.get("response", await response.text())
                print(f"Server response: {response_text}, Time: {time.time() - start:.2f} seconds")
                return response_text
    except Exception as e:
        response_text = f"Server error: {str(e)}"
        print(f"Server error: {response_text}, Time: {time.time() - start:.2f} seconds")
        return response_text

def run_tts(text, component_id):
    """Run TTS in a separate thread and return component ID for highlighting."""
    try:
        print(f"TTS: Speaking '{text}' for component '{component_id}'")
        tts_engine.say(text)
        tts_engine.runAndWait()
        print(f"TTS: Speech completed for component '{component_id}'")
    except Exception as e:
        print(f"TTS error: {str(e)}")
    return component_id

async def speak_results(most_common_emotion, response_text):
    print("Preparing to speak results...")
    start = time.time()
    
    # Reinitialize TTS engine to ensure reusability
    global tts_engine
    try:
        tts_engine.stop()
    except Exception:
        pass
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    
    # Speak emotion
    emotion_text = f"Detected emotion: {most_common_emotion}"
    yield {"emotion_out": "highlighted"}  # Highlight emotion textbox
    tts_thread = threading.Thread(target=run_tts, args=(emotion_text, "emotion_out"), daemon=True)
    tts_thread.start()
    tts_thread.join()
    yield {"emotion_out": ""}  # Remove highlight
    
    # Speak server response
    response_text_safe = response_text or "No response received"
    response_text_to_speak = f"Server response: {response_text_safe}"
    yield {"response_out": "highlighted"}  # Highlight response textbox
    tts_thread = threading.Thread(target=run_tts, args=(response_text_to_speak, "response_out"), daemon=True)
    tts_thread.start()
    tts_thread.join()
    yield {"response_out": ""}  # Remove highlight
    
    print(f"TTS completed, Time: {time.time() - start:.2f} seconds")

async def stop_and_process():
    global recording, audio_path, audio_thread
    print("Stopping recording...")
    if not recording:
        print("No active recording")
        yield "No active recording", "No transcription", "No server response", "Stopping recording...", {}
        return

    recording = False
    if audio_thread:
        audio_thread.join()
        audio_thread = None
    
    try:
        audio_path = audio_queue.get_nowait()
        print(f"Audio saved at: {audio_path}")
    except queue.Empty:
        print("No audio recorded")
        yield "No face detected", "No audio recorded", "No server response", "No audio recorded", {}
        return

    yield "Processing emotions...", "Transcribing audio...", "Awaiting server response...", "Processing...", {}

    most_common_emotion = await process_emotions(snapshots)
    yield most_common_emotion, "Transcribing audio...", "Awaiting server response...", "Processing emotions done", {}

    text = await transcribe_audio(audio_path)
    yield most_common_emotion, text, "Awaiting server response...", "Transcription done", {}

    response_text = await send_to_gemini(text)
    yield most_common_emotion, text, response_text, "Processing complete", {}

    # Speak results and handle highlighting
    async for highlight in speak_results(most_common_emotion, response_text):
        yield most_common_emotion, text, response_text, "Processing complete", highlight

def get_recording_status():
    """Function to manually update recording status."""
    return "Recording audio and video..." if recording else "Not recording"

def cleanup():
    global cap, executor, tts_engine
    if cap and cap.isOpened():
        cap.release()
    executor.shutdown()
    try:
        tts_engine.stop()
    except Exception:
        pass

# JavaScript to toggle highlighting
custom_js = """
function highlight_component(highlight_state) {
    const emotion_out = document.querySelector('#emotion_id textarea');
    const response_out = document.querySelector('#response_text_id textarea');
    
    if (highlight_state.emotion_out === 'highlighted') {
        emotion_out.classList.add('highlighted');
    } else {
        emotion_out.classList.remove('highlighted');
    }
    
    if (highlight_state.response_out === 'highlighted') {
        response_out.classList.add('highlighted');
    } else {
        response_out.classList.remove('highlighted');
    }
}
"""

with gr.Blocks(title="Video Call App", css=custom_css) as demo:
    gr.Markdown("## Video Call App")
    gr.Markdown("Click 'Start Recording' to begin audio and video. Record for 10–15 seconds, speaking clearly, then click 'Stop and Process'. Use 'Refresh Status' to update recording status.")
    image_out = gr.Image(label="Live Webcam")
    status_out = gr.Textbox(label="Recording Status", value="Not recording", interactive=False)
    start_btn = gr.Button("Start Recording")
    stop_btn = gr.Button("Stop and Process")
    refresh_btn = gr.Button("Refresh Status")
    emotion_out = gr.Textbox(label="Detected Emotion", elem_id="emotion_id", interactive=False)
    transcript_out = gr.Textbox(label="Transcription", interactive=False)
    response_out = gr.Textbox(label="Server Response", elem_id="response_text_id", interactive=False)
    start_btn.click(fn=start_recording, outputs=status_out)
    stop_btn.click(fn=stop_and_process, outputs=[emotion_out, transcript_out, response_out, status_out, gr.State()])
    refresh_btn.click(fn=get_recording_status, outputs=status_out)
    demo.load(video_stream, outputs=image_out)
    demo.unload(cleanup)

demo.launch(debug=True)