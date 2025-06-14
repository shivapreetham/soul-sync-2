import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import soundfile as sf
import speech_recognition as sr
from collections import Counter

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks(frame):
    """
    Returns numpy array shape (468,2) of pixel coords, or None if no face.
    """
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

def detect_emotion_from_frame(
    frame,
    thresh_smile=0.035,
    thresh_surprise=0.25,
    thresh_anger=0.05,
    thresh_sad_slope=0.02
) -> str | None:
    """
    Heuristic emotion detection from one frame:
      - happy: MAR > thresh_smile
      - surprised: avg EAR > thresh_surprise
      - angry: eyebrow_distance < thresh_anger * face_width
      - sad: mouth_corner_slope > thresh_sad_slope
      - else: neutral
    Returns emotion or None if no face detected.
    """
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

def process(video_path, audio_data):
    """
    Process the recorded video and audio:
    - Detect the most common emotion from video frames.
    - Transcribe the audio to text.
    """
    # Process video for emotions
    cap = cv2.VideoCapture(video_path)
    emotions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        emotion = detect_emotion_from_frame(frame)
        if emotion:
            emotions.append(emotion)
    cap.release()

    # Determine the most common emotion
    if emotions:
        most_common_emotion = Counter(emotions).most_common(1)[0][0]
    else:
        most_common_emotion = "No face detected"

    # Transcribe audio
    sample_rate, audio_array = audio_data
    audio_file = "temp_audio.wav"
    sf.write(audio_file, audio_array, sample_rate)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Could not understand audio"
    except sr.RequestError:
        text = "Could not request results"

    return video_path, audio_data, most_common_emotion, text

def main():
    with gr.Blocks(title="Emotion and Speech Demo") as demo:
        gr.Markdown("**Record video and audio, then click Submit to process.**")
        with gr.Row():
            video_input = gr.Video(sources="webcam", label="Record a short video")
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Record audio")
        with gr.Row():
            video_output = gr.Video(label="Playback Video")
            audio_output = gr.Audio(label="Playback Audio")
        with gr.Row():
            emotion_output = gr.Textbox(label="Detected Emotion")
            text_output = gr.Textbox(label="Transcribed Text")
        submit = gr.Button("Submit")
        submit.click(
            fn=process,
            inputs=[video_input, audio_input],
            outputs=[video_output, audio_output, emotion_output, text_output]
        )
    demo.launch()

if __name__ == "__main__":
    main()