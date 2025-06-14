
import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import wave
import json
import scipy.io.wavfile as wavfile
import requests
import pyaudio
import threading
import queue
import librosa
from vosk import Model, KaldiRecognizer
from collections import Counter

     # Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
         static_image_mode=True,
         max_num_faces=1,
         refine_landmarks=True,
         min_detection_confidence=0.5
     )

     # Initialize Vosk model
VOSK_MODEL_PATH = "D:/College_Life/projects/Hackathon/Edge AI/soul-sync-2/vosk-model-small-en-us-0.15"
if not os.path.isdir(VOSK_MODEL_PATH):
         raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}. Download and unzip a Vosk model there.")
vosk_model = Model(VOSK_MODEL_PATH)

     # Voice command detection state
command_queue = queue.Queue()
recording_triggered = threading.Event()

def mouth_aspect_ratio(lm):
         left, right = lm[61], lm[291]
         top, bottom = lm[13], lm[14]
    def get_landmarks(frame):
         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         results = face_mesh.process(rgb)
         if not results.multi_face_landmarks:
             return None
         lm = results.multi_face_landmarks[0].landmark
         h, w, _ = frame.shape
         coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
         return coords

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

     def transcribe_with_vosk(audio_path):
         wf = wave.open(audio_path, "rb")
         if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
             wf.close()
             return "[Audio must be mono 16-bit]"
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
         resf = json.loads(rec.FinalResult())
         if "text" in resf:
             text_parts.append(resf["text"])
         wf.close()
         return " ".join(text_parts).strip() or "No speech detected"

     def transcribe_audio_numpy(audio_data):
         if audio_data is None:
             return ""
         sample_rate, arr = audio_data
         if arr.ndim == 2:
             arr = arr.mean(axis=1)
         target_sr = 16000
         if sample_rate != target_sr:
             arr = librosa.resample(arr.astype(float), orig_sr=sample_rate, target_sr=target_sr)
             sample_rate = target_sr
         if arr.dtype.kind == 'f':
             arr16 = (arr * 32767).astype(np.int16)
         else:
             arr16 = arr.astype(np.int16)
         fd, path = tempfile.mkstemp(suffix=".wav")
         os.close(fd)
         wavfile.write(path, sample_rate, arr16)
         try:
             text = transcribe_with_vosk(path)
         except Exception as e:
             text = f"[Vosk error: {e}]"
         finally:
             try:
                 os.remove(path)
             except:
                 pass
         return text

     def extract_emotions_from_video(video_path, frame_interval_sec=1.0):
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

     def pick_dominant_emotion(emotion_counts):
         if not emotion_counts:
             return "no_face_detected"
         if len(emotion_counts) == 1 and "no_face" in emotion_counts:
             return "no_face_detected"
         filtered = {e: c for e, c in emotion_counts.items() if e != "no_face"}
         if not filtered:
             return "no_face_detected"
         return max(filtered.items(), key=lambda x: x[1])[0]

     def send_to_server(text, emotion):
         url = "https://sign-language-3-5vax.onrender.com/gemini"
         payload = {"text": text, "emotion": emotion}
         try:
             resp = requests.post(url, json=payload, timeout=10)
             try:
                 j = resp.json()
                 return j.get("reply", json.dumps(j))
             except:
                 return resp.text
         except Exception as e:
             return f"[Request to server failed: {e}]"

     def listen_for_command():
         p = pyaudio.PyAudio()
         stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
         rec = KaldiRecognizer(vosk_model, 16000)
         print("Listening for 'Start video call'...")
         while not recording_triggered.is_set():
             data = stream.read(4000, exception_on_overflow=False)
             if rec.AcceptWaveform(data):
                 res = json.loads(rec.Result())
                 text = res.get("text", "").lower()
                 if "start video call" in text:
                     print("Command detected: Start video call")
                     command_queue.put("start")
                     recording_triggered.set()
                     break
         stream.stop_stream()
         stream.close()
         p.terminate()

     def process(video_path, audio_data):
         if not video_path or not audio_data:
             return "No video/audio recorded", "", "[No data to process]"
         try:
             counts = extract_emotions_from_video(video_path, frame_interval_sec=1.0)
             dominant = pick_dominant_emotion(counts)
         except Exception as e:
             dominant = f"[Emotion error: {e}]"
         try:
             text = transcribe_audio_numpy(audio_data)
         except Exception as e:
             text = f"[Transcription error: {e}]"
         reply = send_to_server(text, dominant)
         return dominant, text, reply

     def start_recording():
         recording_triggered.clear()
         threading.Thread(target=listen_for_command, daemon=True).start()
         return "Say 'Start video call' to begin recording."

     def main():
         with gr.Blocks(title="Emotion-Aware Video Call") as demo:
             gr.Markdown("## Emotion-Aware Video Call\n"
                         "1. Click **Listen for Command** to start listening for 'Start video call'.\n"
                         "2. Say 'Start video call' to begin recording video and audio.\n"
                         "3. Stop recording, then click **Submit** to process:\n"
                         "   - Detect emotions from video snapshots (1 per second).\n"
                         "   - Transcribe audio with Vosk.\n"
                         "   - Send text and emotion to Gemini server and show response.")
             status = gr.Textbox(label="Status", interactive=False)
             with gr.Row():
                 video_in = gr.Video(sources=["webcam"], label="Record video")
                 audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Record audio")
             with gr.Row():
                 emotion_out = gr.Textbox(label="Detected Dominant Emotion", interactive=False)
                 transcript_out = gr.Textbox(label="Transcription", interactive=False)
             reply_out = gr.Textbox(label="Gemini Server Reply", interactive=False)
             listen_btn = gr.Button("Listen for Command")
             submit_btn = gr.Button("Submit")
             listen_btn.click(fn=start_recording, outputs=status)
             submit_btn.click(fn=process, inputs=[video_in, audio_in], outputs=[emotion_out, transcript_out, reply_out])
         demo.launch()

     if __name__ == "__main__":
         main()
     ```