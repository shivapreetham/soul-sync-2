import re
from typing import List, Tuple, Optional

def build_prompt(
    history: List[Tuple[str, str]],
    user_input: str,
    emotion_label: str = "neutral",
    model_type: str = "",
    tokenizer=None,
    max_history_tokens: int = 600
) -> str:
    """
    Build conversation prompt based on model_type.
    For DialoGPT models, model_type should contain "DialoGPT".
    """
    if "DialoGPT".lower() in model_type.lower():
        return build_dialogpt_prompt(history, user_input, emotion_label, tokenizer)
    else:
        # Generic fallback
        return build_generic_prompt(history, user_input, emotion_label)

def build_dialogpt_prompt(
    history: List[Tuple[str, str]],
    user_input: str,
    emotion_label: str,
    tokenizer=None
) -> str:
    """
    Prompt format for DialoGPT:
    [optional emotion context][history exchanges][current input + eos_token]
    """
    parts: List[str] = []
    # Emotion context
    if emotion_label and emotion_label != "neutral":
        ctx = get_emotion_context(emotion_label)
        if ctx:
            parts.append(ctx)
            parts.append(tokenizer.eos_token if tokenizer and tokenizer.eos_token else "\n")
    # Include last 3 exchanges
    recent = history[-3:] if history else []
    for user_msg, bot_msg in recent:
        parts.append(user_msg)
        if tokenizer and tokenizer.eos_token:
            parts.append(tokenizer.eos_token)
        parts.append(bot_msg)
        if tokenizer and tokenizer.eos_token:
            parts.append(tokenizer.eos_token)
    # Current input
    parts.append(user_input)
    if tokenizer and tokenizer.eos_token:
        parts.append(tokenizer.eos_token)
    # Join without extra separators
    return "".join(parts)

def build_generic_prompt(
    history: List[Tuple[str, str]],
    user_input: str,
    emotion_label: str
) -> str:
    """
    Simple User/Assistant format with optional emotion context.
    """
    parts: List[str] = []
    if emotion_label and emotion_label != "neutral":
        ctx = get_emotion_context(emotion_label)
        if ctx:
            parts.append(f"{ctx}\n\n")
    recent = history[-2:] if history else []
    for user_msg, bot_msg in recent:
        parts.append(f"User: {user_msg}\n")
        parts.append(f"Assistant: {bot_msg}\n")
    parts.append(f"User: {user_input}\n")
    parts.append("Assistant:")
    return "".join(parts)

def get_emotion_context(emotion: str) -> str:
    """
    Return a brief instruction based on emotion.
    """
    contexts = {
        "happy": "The user is feeling happy. Respond with enthusiasm.",
        "excited": "The user is excited. Match their energy.",
        "sad": "The user seems sad. Be empathetic and supportive.",
        "frustrated": "The user appears frustrated. Be patient and helpful.",
        "confused": "The user seems confused. Provide clear explanations.",
        "angry": "The user seems upset. Stay calm and understanding.",
        "anxious": "The user appears anxious. Be reassuring.",
        "tired": "The user seems tired. Be gentle."
    }
    return contexts.get(emotion, "")

def truncate_history(
    history: List[Tuple[str, str]],
    tokenizer=None,
    max_tokens: int = 800
) -> List[Tuple[str, str]]:
    """
    Truncate history to fit within max_tokens.
    If tokenizer is provided, uses token counts; else fallback to char-length heuristic.
    Keeps the most recent exchanges.
    """
    if not history:
        return history
    if not tokenizer:
        # approximate by characters
        max_chars = max_tokens * 4
        total = sum(len(u) + len(b) for u, b in history)
        h = history.copy()
        while total > max_chars and len(h) > 1:
            u0, b0 = h.pop(0)
            total -= (len(u0) + len(b0))
        return h

    # Token-based: accumulate from most recent backwards
    truncated: List[Tuple[str, str]] = []
    total_tokens = 0
    for user_msg, bot_msg in reversed(history):
        text = f"{user_msg} {bot_msg}"
        try:
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            cnt = tokens.input_ids.shape[-1]
        except Exception:
            cnt = len(text) // 4
        if total_tokens + cnt > max_tokens and truncated:
            break
        truncated.append((user_msg, bot_msg))
        total_tokens += cnt
    return list(reversed(truncated))

def validate_response(response: str, user_input: str, emotion: str) -> Tuple[bool, str]:
    """
    Basic cleaning and quality checks on model response.
    Returns (is_valid, cleaned_response).
    """
    if not response:
        return False, response
    text = response.strip()
    if len(text) < 3:
        return False, text
    # Truncate overly long
    if len(text) > 500:
        parts = re.split(r'[.!?]+', text)
        if len(parts) > 1:
            text = parts[0].strip() + "."
    # Check repetition
    words = text.lower().split()
    if len(words) > 3:
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        if max(counts.values()) > len(words) * 0.4:
            return False, text
    # Bad patterns
    low = text.lower()
    bad = [r'^(um+|uh+|er+)\s', r'^\s*\?\s*$', r'^\s*\.+\s*$']
    for p in bad:
        if re.search(p, low):
            return False, text
    # Cleanup punctuation repeats
    text = re.sub(r'([.!?]){3,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return True, text
