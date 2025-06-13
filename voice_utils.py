# # voice_utils.py

# import streamlit as st
# import sounddevice as sd
# import pyttsx3
# import numpy as np
# import tempfile
# import os
# import json

# # Vosk for offline ASR
# try:
#     from vosk import Model, KaldiRecognizer
#     VOSK_AVAILABLE = True
# except ImportError:
#     VOSK_AVAILABLE = False

# class VoiceProcessor:
#     """
#     Offline voice input/output using Vosk (ASR) and pyttsx3 (TTS).
#     Requires a local Vosk model downloaded beforehand.
#     """
#     def __init__(self, model_path: str):
#         if not VOSK_AVAILABLE:
#             raise ImportError(
#                 "Vosk not installed. Install via `pip install vosk sounddevice` "
#                 "and ensure a Vosk model is downloaded."
#             )
#         if not os.path.isdir(model_path):
#             raise FileNotFoundError(
#                 f"Vosk model not found at '{model_path}'.\n"
#                 "Download an offline model from https://alphacephei.com/vosk/models, "
#                 "extract locally, and pass its path."
#             )
#         # Load Vosk model
#         self.model = Model(model_path)
#         self.samplerate = 16000  # ensure model matches this rate
#         # Initialize TTS
#         self.tts_engine = pyttsx3.init()
#         # Choose a default voice (e.g., first available or female if found)
#         voices = self.tts_engine.getProperty('voices')
#         if voices:
#             for v in voices:
#                 if hasattr(v, 'name') and 'female' in v.name.lower():
#                     self.tts_engine.setProperty('voice', v.id)
#                     break
#             else:
#                 self.tts_engine.setProperty('voice', voices[0].id)
#         # Default rate/volume
#         self.tts_engine.setProperty('rate', 200)
#         self.tts_engine.setProperty('volume', 0.9)

#     def record_and_transcribe(self, duration: int = 5) -> str | None:
#         """
#         Record from default microphone for `duration` seconds (offline).
#         Returns transcribed text or None if empty.
#         """
#         if not VOSK_AVAILABLE:
#             st.error("Offline ASR not available.")
#             return None
#         st.info(f"Recording {duration}s...")
#         try:
#             audio = sd.rec(int(duration * self.samplerate), samplerate=self.samplerate,
#                            channels=1, dtype='int16')
#             sd.wait()
#         except Exception as e:
#             st.error(f"Audio recording failed: {e}")
#             return None

#         rec = KaldiRecognizer(self.model, self.samplerate)
#         data = audio.tobytes()
#         if rec.AcceptWaveform(data):
#             res = rec.Result()
#         else:
#             res = rec.FinalResult()
#         try:
#             text = json.loads(res).get('text', '').strip()
#         except Exception:
#             text = None
#         if text:
#             st.success(f"Transcribed: {text}")
#             return text
#         else:
#             st.warning("No speech recognized")
#             return None

#     def text_to_speech(self, text: str, speed: float = 1.0, save_file: bool = False) -> str | None:
#         """
#         Speak `text` via pyttsx3 offline.
#         If save_file=True, saves WAV to temp file and plays in Streamlit.
#         Returns path if saved, else None.
#         """
#         if not text:
#             return None
#         cleaned = self._clean_text_for_speech(text)
#         # Adjust rate
#         base_rate = 200
#         self.tts_engine.setProperty('rate', int(base_rate * speed))
#         if save_file:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#                 path = tmp.name
#             try:
#                 self.tts_engine.save_to_file(cleaned, path)
#                 self.tts_engine.runAndWait()
#                 with open(path, 'rb') as f:
#                     audio_bytes = f.read()
#                     st.audio(audio_bytes, format='audio/wav')
#                 return path
#             except Exception as e:
#                 st.error(f"TTS save/play failed: {e}")
#                 return None
#             finally:
#                 try:
#                     os.unlink(path)
#                 except:
#                     pass
#         else:
#             try:
#                 self.tts_engine.say(cleaned)
#                 self.tts_engine.runAndWait()
#             except Exception as e:
#                 st.error(f"TTS failed: {e}")
#             return None

#     def _clean_text_for_speech(self, text: str) -> str:
#         import re
#         # remove markdown/code formatting
#         text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
#         text = re.sub(r'\*(.*?)\*', r'\1', text)
#         text = re.sub(r'`(.*?)`', r'\1', text)
#         # common expansions
#         repl = {
#             'AI': 'artificial intelligence',
#             'UI': 'user interface',
#             'API': 'A P I',
#             'URL': 'U R L',
#             'HTML': 'H T M L',
#             'CSS': 'C S S',
#             'JS': 'JavaScript',
#             'etc.': 'etcetera',
#             'e.g.': 'for example',
#             'i.e.': 'that is',
#         }
#         for k, v in repl.items():
#             text = text.replace(k, v)
#         text = re.sub(r'(\d+)%', r'\1 percent', text)
#         text = re.sub(r'(\d+)\+', r'\1 plus', text)
#         text = re.sub(r'&', ' and ', text)
#         return re.sub(r'\s+', ' ', text).strip()

#     def test_microphone(self) -> bool:
#         """Quick test recording 1 second."""
#         try:
#             sd.rec(int(1 * self.samplerate), samplerate=self.samplerate,
#                    channels=1, dtype='int16')
#             sd.wait(timeout=2)
#             return True
#         except:
#             return False

# # Simple fallback using system TTS only (no ASR)
# class SimpleVoiceProcessor:
#     def __init__(self):
#         self.supported = False
#         try:
#             import platform
#             import subprocess
#             self.system = platform.system()
#             self.supported = True
#         except:
#             pass

#     def record_and_transcribe(self, duration: int = 5) -> str | None:
#         st.warning("Offline ASR not available in SimpleVoiceProcessor.")
#         return None

#     def text_to_speech(self, text: str, speed: float = 1.0) -> None:
#         if not text:
#             return
#         import platform, os
#         # basic system TTS
#         system = platform.system()
#         if system == "Darwin":
#             os.system(f'say "{text}"')
#         elif system == "Windows":
#             os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; '
#                       f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
#         elif system == "Linux":
#             os.system(f'espeak "{text}"')

# def get_voice_processor(vosk_model_path: str = None):
#     """
#     Returns VoiceProcessor configured with local Vosk model.
#     Pass the local model directory to vosk_model_path.
#     """
#     if vosk_model_path:
#         try:
#             return VoiceProcessor(vosk_model_path)
#         except Exception as e:
#             st.warning(f"VoiceProcessor init failed: {e}")
#     # Fallback
#     proc = SimpleVoiceProcessor()
#     if proc.supported:
#         return proc
#     raise ImportError(
#         "No offline voice capability. To enable ASR, install vosk and download a model."
#     )
