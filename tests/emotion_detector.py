# emotion_detector.py

import cv2 
import mediapipe as mp
import numpy as np

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

def run_webcam_emotion():
    """
    Demo mode: opens webcam and overlays detected emotion live.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion_from_frame(frame) or "No face"
        cv2.putText(frame, emotion, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_emotion()
