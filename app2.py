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
vosk_model = Model("model")

# Global variables for recording state and data storage
recording = False
snapshots = []
audio_path = None
audio_thread = None
audio_queue = queue.Queue()

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
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

def detect_emotion_from_frame(frame, thresh_smile=0.035, thresh_surprise=0.25, thresh_anger=0.05, thresh_sad_slope=0.02):
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

def record_audio():
    global audio_path
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
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

def capture_snapshots():
    global snapshots
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open webcam"
    last_time = time.time()
    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        if current_time - last_time >= 1.0:  # Capture one snapshot per second
            snapshots.append(frame.copy())
            last_time = current_time
    cap.release()

def start_recording():
    global recording, snapshots, audio_thread
    if not recording:
        recording = True
        snapshots = []
        threading.Thread(target=capture_snapshots, daemon=True).start()
        audio_thread = threading.Thread(target=record_audio, daemon=True)
        audio_thread.start()
        return "Recording started..."
    return "Already recording"

def stop_and_process():
    global recording, audio_path
    if not recording:
        return "Not recording"
    recording = False
    if audio_thread:
        audio_thread.join()
    try:
        audio_path = audio_queue.get_nowait()
    except queue.Empty:
        return "Error: No audio recorded"

    # Process emotions
    emotions = [detect_emotion_from_frame(frame) for frame in snapshots]
    emotions = [e for e in emotions if e is not None]
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "unknown"

    # Transcribe audio
    data, samplerate = sf.read(audio_path)
    recognizer = KaldiRecognizer(vosk_model, samplerate)
    recognizer.AcceptWaveform(data.tobytes())
    result = json.loads(recognizer.Result())
    text = result.get("text", "")

    # Send to Gemini server
    payload = {"emotion": most_common_emotion, "text": text}
    try:
        response = requests.post("http://localhost:8000/process", json=payload, timeout=5)
        response_text = response.json().get("response", "No response")
    except Exception as e:
        response_text = f"Error: {str(e)}"

    return response_text

with gr.Blocks(title="Video Call App") as demo:
    gr.Markdown("## Video Call App")
    gr.Markdown("Click 'Start Video Call' to begin recording. Click 'Stop and Process' to get the response.")
    start_btn = gr.Button("Start Video Call")
    stop_btn = gr.Button("Stop and Process")
    output_text = gr.Textbox(label="Output")
    start_btn.click(fn=start_recording, outputs=output_text)
    stop_btn.click(fn=stop_and_process, outputs=output_text)

demo.launch()