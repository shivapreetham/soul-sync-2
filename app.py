# # main_app.py

# import streamlit as st
# import torch
# import time
# from datetime import datetime

# # Existing utilities; adjust import paths if needed
# from llm_utils2 import load_model, generate_response, get_smart_fallback
# from chat_utils2 import build_prompt, truncate_history, validate_response
# from voice_utils import get_voice_processor

# # Emotion detector: ensure module name matches your file
# from emotion_classifier import detect_emotion_from_frame

# # For video+audio streaming
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# import av
# import numpy as np
# import soundfile as sf
# import tempfile
# import os
# import speech_recognition as sr

# st.set_page_config(
#     page_title="Soul Sync Chat v2.1",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # --- Helper functions defined first ---

# def detect_emotion_from_text(text: str) -> str:
#     """Simple keyword-based emotion detection."""
#     t = text.lower()
#     if any(w in t for w in ["happy","great","awesome","love","excited"]):
#         return "happy"
#     if any(w in t for w in ["sad","upset","hurt","cry","terrible"]):
#         return "sad"
#     if any(w in t for w in ["angry","mad","hate","frustrated","annoyed"]):
#         return "angry"
#     if any(w in t for w in ["confused","don't understand","unclear","lost"]):
#         return "confused"
#     if text.count('!') >= 2:
#         return "excited"
#     if text.count('?') >= 2:
#         return "confused"
#     return "neutral"

# def export_chat() -> str:
#     """Export chat history as plain text."""
#     lines = []
#     lines.append(f"Soul Sync Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     lines.append("="*50)
#     for i, (u,b,emotion,timing) in enumerate(st.session_state.messages, 1):
#         lines.append(f"\nExchange {i}:")
#         lines.append(f"User ({emotion}): {u}")
#         lines.append(f"Bot: {b}")
#         lines.append(f"Response time: {timing.get('total_time',0):.2f}s")
#         lines.append("-"*30)
#     return "\n".join(lines)

# def process_message(
#     user_input: str,
#     voice_enabled: bool,
#     voice_processor,
#     voice_speed: float,
#     tokenizer,
#     model,
#     device,
#     max_tokens: int,
#     temperature: float,
#     top_p: float,
#     top_k: int,
#     debug_mode: bool,
#     model_name: str,
#     external_emotion: str = None
# ):
#     """Process user message, possibly with external_emotion from video capture."""
#     start_time = time.time()
#     timing = {}

#     # Determine emotion: prefer external_emotion if provided
#     if external_emotion:
#         emotion = external_emotion
#     else:
#         emotion = detect_emotion_from_text(user_input)

#     # Build prompt
#     with st.spinner("Thinking..."):
#         t0 = time.time()
#         history = st.session_state.conversation_history if use_history else []
#         prompt = build_prompt(history, user_input, emotion_label=emotion, model_type=model_name, tokenizer=tokenizer)
#         timing['prompt_time'] = time.time() - t0

#         # Generate response
#         t1 = time.time()
#         response = generate_response(
#             prompt=prompt,
#             tokenizer=tokenizer,
#             model=model,
#             device=device,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             history=history,
#             user_input=user_input,
#             emotion=emotion
#         )
#         timing['inference_time'] = time.time() - t1

#         # Validate
#         t2 = time.time()
#         is_valid, final_response = validate_response(response, user_input, emotion)
#         if not is_valid or not final_response:
#             final_response = get_smart_fallback(user_input, emotion)
#         timing['cleaning_time'] = time.time() - t2

#     # Voice output
#     if voice_enabled and voice_processor and final_response:
#         try:
#             with st.spinner("Speaking..."):
#                 voice_processor.text_to_speech(final_response, speed=voice_speed)
#         except Exception as e:
#             if debug_mode:
#                 st.error(f"Voice output failed: {e}")

#     timing['total_time'] = time.time() - start_time

#     # Update history
#     st.session_state.conversation_history.append((user_input, final_response))
#     st.session_state.messages.append((user_input, final_response, emotion, timing))

#     # Truncate if too long
#     st.session_state.conversation_history = truncate_history(
#         st.session_state.conversation_history,
#         tokenizer,
#         max_tokens=800
#     )

#     # Rerun to display updated chat
#     st.rerun()


# # --- Sidebar and settings ---

# st.sidebar.header("🎛️ Configuration")

# # Only DialoGPT-medium and DialoGPT-large
# model_options = {
#     "DialoGPT Medium (355M)": "microsoft/DialoGPT-medium",
#     "DialoGPT Large (774M)": "microsoft/DialoGPT-large"
# }
# selected_model_label = st.sidebar.selectbox(
#     "Choose Model", list(model_options.keys()), index=0
# )
# model_name = model_options[selected_model_label]

# # Context/history toggle
# use_history = st.sidebar.checkbox("Include chat history in prompt?", value=True)

# # Debug mode
# debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)
# st.session_state.debug_mode = debug_mode

# # Voice settings
# st.sidebar.header("🎤 Voice Settings")
# try:
#     voice_processor = get_voice_processor()  # may require Vosk model path if using offline ASR
#     voice_enabled = st.sidebar.checkbox("Enable Voice Input/Output", value=False)
#     if voice_enabled:
#         voice_speed = st.sidebar.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)
#     else:
#         voice_speed = 1.0
# except Exception:
#     st.sidebar.warning("Voice not available: install/configure dependencies")
#     voice_processor = None
#     voice_enabled = False
#     voice_speed = 1.0

# # Video settings
# st.sidebar.header("📹 Video Settings")
# video_enabled = st.sidebar.checkbox("Enable Video Emotion Capture", value=False)
# if video_enabled:
#     st.sidebar.markdown(
#         "- Click **Start Video** below to begin webcam emotion capture.\n"
#         "- Speak while video is on; click **Done Recording** when finished."
#     )

# # Generation settings
# st.sidebar.header("⚙️ Generation Settings")
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     temperature = st.slider("Temperature", 0.6, 1.2, 0.8, 0.1)
#     top_k = st.slider("Top-k", 30, 100, 50, 10)
# with col2:
#     top_p = st.slider("Top-p", 0.7, 0.95, 0.9, 0.05)
#     max_tokens = st.number_input("Max tokens", 20, 150, 50, 10)

# # Load model with caching
# @st.cache_resource
# def init_model(name: str):
#     return load_model(name)

# try:
#     with st.spinner(f"Loading {selected_model_label}..."):
#         tokenizer, model, device = init_model(model_name)
#     st.sidebar.success(f"Model loaded on {device}")
#     if torch.cuda.is_available():
#         mem = torch.cuda.memory_allocated() / 1024**3
#         st.sidebar.info(f"GPU Memory: {mem:.1f} GB")
# except Exception as e:
#     st.sidebar.error(f"Failed to load model: {e}")
#     if debug_mode:
#         st.exception(e)
#     st.stop()

# # Session state init
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "conversation_history" not in st.session_state:
#     st.session_state.conversation_history = []

# # For video/audio capture accumulation
# if "video_frames" not in st.session_state:
#     st.session_state.video_frames = []
# if "audio_frames" not in st.session_state:
#     st.session_state.audio_frames = []
# if "webrtc_ctx" not in st.session_state:
#     st.session_state.webrtc_ctx = None
# if "recording" not in st.session_state:
#     st.session_state.recording = False

# # --- Main UI ---

# st.title("🤖 Soul Sync Chat")
# st.markdown("*Emotion-aware chatbot with voice & video cue capture*")

# # Display chat history
# st.header("💬 Conversation")
# chat_container = st.container()
# with chat_container:
#     for (user_msg, bot_msg, emotion, timing) in st.session_state.messages:
#         # User
#         with st.chat_message("user"):
#             emoji = {
#                 "happy":"😊","sad":"😢","angry":"😠","frustrated":"😤",
#                 "confused":"🤔","excited":"🤩","anxious":"😰","tired":"😴"
#             }.get(emotion, "💬")
#             st.write(f"{emoji} {user_msg}")
#             if debug_mode:
#                 st.caption(f"Emotion: {emotion} | Time: {timing.get('total_time',0):.2f}s")
#         # Bot
#         with st.chat_message("assistant"):
#             st.write(bot_msg)

# # --- Video + Audio capture section ---
# if video_enabled:
#     st.subheader("📹 Video & Audio Input")

#     class EmotionProcessor(VideoProcessorBase):
#         def __init__(self):
#             self.frame_count = 0

#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             img = frame.to_ndarray(format="bgr24")
#             emotion = detect_emotion_from_frame(img)
#             label = emotion or "No face"
#             import cv2
#             cv2.putText(img, label, (30,30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#             st.session_state.video_frames.append(img.copy())
#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     col_start, col_done = st.columns(2)
#     if col_start.button("▶️ Start Video"):
#         st.session_state.video_frames = []
#         st.session_state.audio_frames = []
#         st.session_state.recording = True
#         ctx = webrtc_streamer(
#             key="emotion-webrtc",
#             mode=WebRtcMode.SENDRECV,
#             video_processor_factory=EmotionProcessor,
#             rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
#             media_stream_constraints={"video": True, "audio": True},
#             async_processing=True,
#         )
#         st.session_state.webrtc_ctx = ctx

#     if col_done.button("⏹️ Done Recording"):
#         st.session_state.recording = False
#         ctx = st.session_state.webrtc_ctx
#         if ctx:
#             ctx.stop()
#         # Process video frames for dominant emotion
#         emotions = []
#         for frame in st.session_state.video_frames:
#             em = detect_emotion_from_frame(frame)
#             if em:
#                 emotions.append(em)
#         dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
#         st.session_state.captured_emotion = dominant_emotion
#         st.success(f"Captured dominant emotion: {dominant_emotion}")

#         # Process audio via SpeechRecognition
#         transcription = ""
#         if ctx and ctx.audio_receiver:
#             audio_chunks = []
#             while True:
#                 try:
#                     audio_frame = ctx.audio_receiver.recv(timeout=1.0)
#                     arr = audio_frame.to_ndarray()
#                     audio_chunks.append(arr)
#                 except Exception:
#                     break
#             st.session_state.audio_frames = audio_chunks

#             if st.session_state.audio_frames:
#                 try:
#                     # Assume sample rate 48000 if unknown
#                     sample_rate = ctx.audio_receiver.buffer[0].format.sample_rate \
#                                   if ctx.audio_receiver and ctx.audio_receiver.buffer else 48000
#                     # Convert to mono
#                     mono = []
#                     for arr in st.session_state.audio_frames:
#                         if arr.ndim > 1:
#                             mono_frame = arr.mean(axis=0)
#                         else:
#                             mono_frame = arr
#                         mono.append(mono_frame)
#                     full_audio = np.concatenate(mono)
#                     tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#                     sf.write(tmp_wav.name, full_audio, sample_rate)
#                     recognizer = sr.Recognizer()
#                     with sr.AudioFile(tmp_wav.name) as source:
#                         audio_data = recognizer.record(source)
#                     try:
#                         transcription = recognizer.recognize_google(audio_data)
#                         st.success("Transcribed speech via Google")
#                     except Exception:
#                         try:
#                             transcription = recognizer.recognize_sphinx(audio_data)
#                             st.success("Transcribed speech via Sphinx")
#                         except Exception:
#                             st.warning("Could not transcribe audio")
#                     finally:
#                         tmp_wav.close()
#                         os.unlink(tmp_wav.name)
#                 except Exception as e:
#                     if debug_mode:
#                         st.error(f"Audio transcription failed: {e}")
#         else:
#             st.info("No audio captured")

#         st.session_state.captured_transcription = transcription
#         if transcription:
#             st.write(f"Transcription: {transcription}")
#             process_message(
#                 transcription,
#                 voice_enabled, voice_processor, voice_speed,
#                 tokenizer, model, device,
#                 max_tokens, temperature, top_p, top_k,
#                 debug_mode, model_name,
#                 external_emotion=st.session_state.captured_emotion
#             )

# # --- Chat input section ---
# if not video_enabled or not st.session_state.get("recording", False):
#     if voice_enabled:
#         st.subheader("🎤 Voice Input")
#         col1, col2 = st.columns([1,1])
#         with col1:
#             if st.button("Start Voice Recording"):
#                 try:
#                     audio_text = voice_processor.record_and_transcribe()
#                     if audio_text:
#                         st.session_state.temp_voice_input = audio_text
#                         st.success(f"Transcribed: {audio_text}")
#                     else:
#                         st.warning("No speech detected")
#                 except Exception as e:
#                     st.error(f"Recording failed: {e}")
#         with col2:
#             if st.button("Use Voice Input"):
#                 if st.session_state.get("temp_voice_input"):
#                     user_input = st.session_state.pop("temp_voice_input")
#                     process_message(
#                         user_input,
#                         voice_enabled, voice_processor, voice_speed,
#                         tokenizer, model, device,
#                         max_tokens, temperature, top_p, top_k,
#                         debug_mode, model_name
#                     )
#     if user_input := st.chat_input("Type your message here..."):
#         process_message(
#             user_input,
#             voice_enabled, voice_processor, voice_speed,
#             tokenizer, model, device,
#             max_tokens, temperature, top_p, top_k,
#             debug_mode, model_name
#         )

# # Sidebar stats & controls
# if st.session_state.messages:
#     st.sidebar.header("📊 Session Stats")
#     num = len(st.session_state.messages)
#     emotions = [m[2] for m in st.session_state.messages]
#     dominant = max(set(emotions), key=emotions.count) if emotions else "neutral"
#     times = [m[3].get('total_time',0) for m in st.session_state.messages]
#     avg_t = sum(times)/len(times) if times else 0
#     st.sidebar.metric("Messages", num)
#     st.sidebar.metric("Dominant Emotion", dominant.title())
#     st.sidebar.metric("Avg Response Time", f"{avg_t:.2f}s")

# st.sidebar.header("🔧 Controls")
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     if st.button("🗑️ Clear Chat"):
#         st.session_state.messages = []
#         st.session_state.conversation_history = []
#         st.rerun()
# with col2:
#     if st.button("💾 Export Chat"):
#         if st.session_state.messages:
#             txt = export_chat()
#             st.download_button("Download Chat", txt, "soul_sync_chat.txt", "text/plain")
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Soul Sync v2.1**")
