# # llm_utils2.py

# import time
# import torch
# import re
# import random
# import streamlit as st
# from packaging import version
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# def load_model(model_name="microsoft/DialoGPT-medium", device=None):
#     """
#     Load a conversational model with optional 4-bit quantization if CUDA is available.
#     For DialoGPT models, avoid trust_remote_code to use built-in GPT2LMHeadModel.
#     Requires torch>=2.6 and transformers>=4.x.
#     Returns (tokenizer, model, device).
#     """
#     # Version checks
#     if version.parse(torch.__version__) < version.parse("2.6.0"):
#         st.error(f"Detected torch v{torch.__version__} < 2.6.0. Please upgrade: pip install --upgrade torch")
#         raise RuntimeError(f"torch>=2.6 required; found {torch.__version__}")
#     tf_ver = transformers.__version__
#     if version.parse(tf_ver) < version.parse("4.0.0"):
#         st.error(f"Detected transformers v{tf_ver} < 4.0.0. Please upgrade: pip install --upgrade transformers")
#         raise RuntimeError(f"transformers>=4.x required; found {tf_ver}")

#     start_time = time.time()
#     st.info(f"🔄 Loading {model_name}...")
#     quantization_config = None
#     use_quant = False

#     # Setup quantization for large models on CUDA
#     if torch.cuda.is_available() and any(k in model_name.lower() for k in ["large", "1.1b", "tinyllama"]):
#         try:
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4"
#             )
#             use_quant = True
#             st.info("🎯 Using 4-bit quantization")
#         except Exception as e:
#             st.warning(f"Quantization unavailable: {e}")

#     # Determine if this is a DialoGPT model
#     is_dialogpt = "microsoft/dialogpt" in model_name.lower() or "microsoft/dialoGPT".lower() in model_name.lower()

#     # Load tokenizer
#     t0 = time.time()
#     try:
#         if is_dialogpt:
#             tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
#                                                       use_fast=True)
#     except Exception as e:
#         st.warning(f"Tokenizer load failed ({e}), retrying without fast:")
#         try:
#             if is_dialogpt:
#                 tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
#                                                           use_fast=False)
#         except Exception as e2:
#             st.error(f"Failed to load tokenizer for '{model_name}': {e2}\n"
#                      "Check model name and internet connection, ensure transformers>=4.x.")
#             raise
#     t_token = time.time() - t0

#     # Prepare model kwargs
#     model_kwargs = {"low_cpu_mem_usage": True}
#     if torch.cuda.is_available():
#         model_kwargs["torch_dtype"] = torch.float16
#     else:
#         model_kwargs["torch_dtype"] = torch.float32

#     if use_quant:
#         model_kwargs["quantization_config"] = quantization_config
#         model_kwargs["device_map"] = "auto"
#     elif torch.cuda.is_available():
#         model_kwargs["device_map"] = "auto"

#     # Load model
#     t1 = time.time()
#     try:
#         if is_dialogpt:
#             model = AutoModelForCausalLM.from_pretrained(model_name)
#         else:
#             model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
#     except Exception as e:
#         st.error(f"Failed to load model '{model_name}': {e}\n"
#                  "Possible causes: wrong model identifier or incompatible transformers version.")
#         raise
#     t_model = time.time() - t1

#     # Ensure pad token exists
#     if tokenizer.pad_token is None:
#         if tokenizer.eos_token:
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             model.resize_token_embeddings(len(tokenizer))
#     if hasattr(model.config, 'pad_token_id'):
#         model.config.pad_token_id = tokenizer.pad_token_id

#     # Device assignment
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if not use_quant and not torch.cuda.is_available():
#         model.to(device)
#     model.eval()

#     total = time.time() - start_time
#     st.success(f"✅ Model loaded on {device}")
#     st.info(f"⏱️ Total load: {total:.2f}s (tokenizer: {t_token:.2f}s, model: {t_model:.2f}s)")
#     if torch.cuda.is_available():
#         mem_alloc = torch.cuda.memory_allocated() / 1024**3
#         mem_res = torch.cuda.memory_reserved() / 1024**3
#         st.info(f"🎯 GPU mem allocated: {mem_alloc:.2f} GB, reserved: {mem_res:.2f} GB")
#     return tokenizer, model, device


# def clean_response(text: str, user_input: str = "") -> str:
#     """
#     Remove special tokens, stop sequences, repetitions, and format nicely.
#     """
#     if not text:
#         return ""
#     text = text.replace("<|endoftext|>", "")
#     text = re.sub(r'^.*?:', '', text)
#     stops = ["Human:", "Assistant:", "User:", "AI:", "Bot:",
#              "\nHuman", "\nUser", "\nAssistant", "\nAI", "\nBot",
#              "<|endoftext|>", "</s>", "<s>", "[INST]", "[/INST]"]
#     for seq in stops:
#         if seq in text:
#             text = text.split(seq)[0]
#     text = re.sub(r'(.{10,}?)\1{2,}', r'\1', text)
#     text = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', text)
#     text = re.sub(r'([.!?])\1{2,}', r'\1', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = re.sub(r'^[\s\.,!?;:\-]+', '', text).strip()
#     if len(text) < 3 or len(text) > 500:
#         return ""
#     if text and not text[-1] in '.!?' and len(text) > 50:
#         parts = re.split(r'[.!?]+', text)
#         if len(parts) > 1 and len(parts[0].strip()) > 20:
#             text = parts[0].strip() + '.'
#     return text


# def apply_emotion_context(prompt: str, emotion: str, user_input: str, tokenizer) -> str:
#     """
#     Prepend or reformat prompt based on detected emotion.
#     """
#     if not emotion or emotion == "neutral":
#         return prompt
#     ctx_map = {
#         "happy": "Respond with enthusiasm and positivity.",
#         "sad": "Be empathetic and supportive.",
#         "frustrated": "Be patient and helpful.",
#         "confused": "Provide clear explanations.",
#         "angry": "Stay calm and understanding.",
#         "anxious": "Be reassuring and supportive.",
#         "excited": "Match their energy appropriately.",
#         "tired": "Be gentle and understanding."
#     }
#     if emotion not in ctx_map:
#         return prompt
#     context = ctx_map[emotion]
#     model_id = getattr(tokenizer, "name_or_path", "")
#     if "dialogpt" in model_id.lower() or "dialoggpt" in model_id.lower():
#         return f"{context} User: {user_input}"
#     else:
#         return f"{context}\n\n{prompt}"


# def generate_response(
#     prompt: str,
#     tokenizer,
#     model,
#     device: torch.device,
#     max_new_tokens: int = 50,
#     temperature: float = 0.8,
#     top_p: float = 0.9,
#     top_k: int = 50,
#     do_sample: bool = True,
#     history=None,
#     user_input: str = "",
#     emotion: str = "neutral"
# ) -> str:
#     """
#     Generate a response. Detect model type from tokenizer.
#     """
#     try:
#         prompt_mod = apply_emotion_context(prompt, emotion, user_input, tokenizer)
#         inputs = tokenizer(
#             prompt_mod,
#             return_tensors="pt",
#             truncation=True,
#             max_length=1024,
#             padding=True
#         )
#         input_ids = inputs.input_ids.to(device)
#         attention_mask = inputs.attention_mask.to(device)
#         seq_len = input_ids.shape[-1]

#         gen_kwargs = {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "top_p": top_p,
#             "top_k": top_k,
#             "do_sample": do_sample,
#             "pad_token_id": tokenizer.pad_token_id,
#             "eos_token_id": tokenizer.eos_token_id,
#             "repetition_penalty": 1.2,
#             "no_repeat_ngram_size": 2,
#             "use_cache": True
#         }
#         model_id = getattr(tokenizer, "name_or_path", "")
#         if "dialogpt" in model_id.lower() or "dialoggpt" in model_id.lower():
#             gen_kwargs.update({
#                 "repetition_penalty": 1.2,
#                 "no_repeat_ngram_size": 3,
#                 "min_length": seq_len + 5
#             })
#         with torch.no_grad():
#             output = model.generate(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 max_length=min(seq_len + max_new_tokens, 1200),
#                 **gen_kwargs
#             )
#         new_tokens = output[0][seq_len:]
#         text = tokenizer.decode(new_tokens, skip_special_tokens=True)
#         return clean_response(text, user_input)
#     except Exception as e:
#         st.error(f"Generation error: {e}")
#         return ""


# def get_smart_fallback(user_input: str, emotion: str = "neutral") -> str:
#     """
#     Simple fallback when model response is invalid or empty.
#     """
#     ui = user_input.lower().strip()
#     if any(w in ui for w in ["hi", "hello", "hey"]):
#         return "Hello! How can I assist you today?"
#     if ui.startswith(("what", "who", "when", "where", "why", "how")):
#         return "That's a good question—could you clarify what you mean?"
#     em_map = {
#         "happy": "I'm glad you're feeling good! Tell me more.",
#         "sad": "I'm sorry you're feeling down. Would you like to talk about it?",
#         "frustrated": "I understand the frustration. How can I help?",
#         "confused": "I see you're confused. What part shall I clarify?",
#         "angry": "I sense you're upset. I'm here to listen.",
#         "excited": "I can feel your excitement! What's on your mind?",
#         "anxious": "I understand you may be worried. How can I reassure you?",
#         "tired": "You seem tired. Would you like to rest or discuss something light?"
#     }
#     if emotion in em_map:
#         return em_map[emotion]
#     defaults = [
#         "Interesting—tell me more.",
#         "Could you elaborate on that?",
#         "What do you think about this further?",
#         "Let's explore that idea more."
#     ]
#     return random.choice(defaults)
