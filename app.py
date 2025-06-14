import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import tempfile, os, wave, json
import scipy.io.wavfile as wavfile
import requests
from vosk import Model, KaldiRecognizer

# -------------- Emotion detector (from emotion_detector.py) --------------
mp_face = mp.solutions.face_mesh
# For frame-based detection we set static_image_mode=False, but since we process individual frames, static_image_mode=True also works.
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

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

def detect_emotion_from_frame(frame,
                              thresh_smile=0.035,
                              thresh_surprise=0.25,
                              thresh_anger=0.05,
                              thresh_sad_slope=0.02) -> str | None:
    lm = get_landmarks(frame)
    if lm is None:
        return None
    # Happy
    if mouth_aspect_ratio(lm) > thresh_smile:
        return "happy"
    # Surprised
    ear = (eye_aspect_ratio(lm, True) + eye_aspect_ratio(lm, False)) / 2.0
    if ear > thresh_surprise:
        return "surprised"
    # Angry
    face_width = np.linalg.norm(lm[234] - lm[454])
    if face_width > 0 and eyebrow_distance(lm) < thresh_anger * face_width:
        return "angry"
    # Sad
    if mouth_corner_slope(lm) > thresh_sad_slope:
        return "sad"
    return "neutral"

# -------------- Vosk ASR setup --------------
# Make sure you have downloaded a Vosk model, e.g.:
#   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
#   unzip to ./models/vosk-model-small-en-us-0.15
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # adjust path as needed
if not os.path.isdir(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}. Download and unzip a Vosk model there.")
vosk_model = Model(VOSK_MODEL_PATH)

def transcribe_with_vosk(audio_path: str) -> str:
    """
    Given a WAV file path, run Vosk ASR and return the transcription.
    Assumes WAV is PCM 16-bit mono.
    """
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        # Need 16-bit mono PCM. If not, we could convert—but for simplicity, require WAV from numpy to be mono 16kHz.
        # Alternatively, convert here with ffmpeg or so, but we assume sample rate 16000 and mono.
        pass
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(False)
    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "text" in res:
                text_parts.append(res["text"])
    # final
    resf = json.loads(rec.FinalResult())
    if "text" in resf:
        text_parts.append(resf["text"])
    wf.close()
    return " ".join(text_parts).strip()

def transcribe_audio_numpy(audio_data):
    """
    audio_data: (sample_rate, np.ndarray)
    We write to a temp WAV with 16-bit PCM, mono, resample to 16000 if needed.
    """
    if audio_data is None:
        return ""
    sample_rate, arr = audio_data  # arr: shape (n,) or (n,2)
    # If stereo, convert to mono by averaging channels
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    # Resample to 16000 if needed
    target_sr = 16000
    if sample_rate != target_sr:
        try:
            import librosa
            arr = librosa.resample(arr.astype(float), orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        except ImportError:
            # If librosa not installed, fallback: skip resample (may reduce accuracy)
            pass
    # Convert float to int16 if needed
    if arr.dtype.kind == 'f':
        arr16 = (arr * 32767).astype(np.int16)
    else:
        arr16 = arr.astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wavfile.write(path, sample_rate, arr16)
    # Transcribe
    try:
        text = transcribe_with_vosk(path)
    except Exception as e:
        text = f"[Vosk ASR error: {e}]"
    finally:
        try: os.remove(path)
        except: pass
    return text

# -------------- Video frame extraction & emotion aggregation --------------
def extract_emotions_from_video(video_path: str, frame_interval_sec: float = 1.0):
    """
    OpenCV read video, sample one frame every frame_interval_sec, detect emotion.
    Return dict emotion->count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = int(fps * frame_interval_sec)
    if step < 1:
        step = 1
    emotion_counts = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            emo = detect_emotion_from_frame(frame) or "no_face"
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        idx += 1
    cap.release()
    return emotion_counts

def pick_dominant_emotion(emotion_counts: dict) -> str:
    if not emotion_counts:
        return "no_face_detected"
    # If only no_face
    if len(emotion_counts) == 1 and "no_face" in emotion_counts:
        return "no_face_detected"
    # Exclude no_face if others exist
    filtered = {e: c for e, c in emotion_counts.items() if e != "no_face"}
    if not filtered:
        return "no_face_detected"
    return max(filtered.items(), key=lambda x: x[1])[0]

# -------------- Send to local Gemini server --------------
def send_to_server(text: str, emotion: str) -> str:
    """
    Sends JSON {"text":..., "emotion":...} to localhost:8000 via POST.
    Expects server returns JSON with e.g. {"reply": "..."} or plain text.
    """
    url = "http://localhost:8000"
    payload = {"text": text, "emotion": emotion}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        # Try to parse JSON
        try:
            j = resp.json()
            # assume reply field
            if isinstance(j, dict) and "reply" in j:
                return j["reply"]
            else:
                # return entire JSON as string
                return json.dumps(j)
        except Exception:
            # fallback to plain text
            return resp.text
    except Exception as e:
        return f"[Request to server failed: {e}]"

# -------------- Main processing function --------------
def process(video_path, audio_data):
    """
    1. Extract frames every 1s from video_path, detect emotions, pick dominant.
    2. Transcribe audio_data via Vosk.
    3. Send to local server and return reply.
    """
    # 1. Emotion
    try:
        counts = extract_emotions_from_video(video_path, frame_interval_sec=1.0)
        dominant = pick_dominant_emotion(counts)
    except Exception as e:
        dominant = f"[Emotion error: {e}]"
    # 2. Transcription
    try:
        text = transcribe_audio_numpy(audio_data)
    except Exception as e:
        text = f"[Transcription error: {e}]"
    # 3. Send to server
    reply = send_to_server(text, dominant)
    return dominant, text, reply

# -------------- Gradio UI --------------
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Emotion-aware capture + send to local server\n"
                    "Click **Start Recording** to record a short clip (video+audio). After stopping, we:\n"
                    "1. Extract snapshots (one frame per second) and detect emotion.\n"
                    "2. Transcribe audio via Vosk.\n"
                    "3. Send `{text, emotion}` to `http://localhost:8000` and display response.\n")
        with gr.Row():
            video_in = gr.Video(source="webcam", label="Record video (will auto-capture snapshots)")
            audio_in = gr.Audio(source="microphone", type="numpy", label="Record audio")
        emotion_out = gr.Textbox(label="Detected Dominant Emotion", interactive=False)
        transcript_out = gr.Textbox(label="Transcription", interactive=False)
        reply_out = gr.Textbox(label="Server Reply", interactive=False)
        submit = gr.Button("Submit")
        submit.click(fn=process,
                     inputs=[video_in, audio_in],
                     outputs=[emotion_out, transcript_out, reply_out])
    demo.launch()

if __name__ == "__main__":
    main()
