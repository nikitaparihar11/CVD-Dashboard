# ===============================
# LANGUAGE ENGINE (Hindi / English / Hinglish)
# ===============================

from langdetect import detect

# ===============================
# TRANSLATIONS
# ===============================
TRANSLATIONS = {
    "welcome_title": {
        "en": "💬 AI Cardiovascular Risk Assistant",
        "hi": "💬 AI हृदय जोखिम सहायक",
        "hi-en": "💬 AI Cardiovascular Risk Assistant (Hinglish Mode)"
    },
    "disclaimer": {
        "en": "⚠️ Disclaimer: I am an AI assistant, not a medical doctor.\nAlways consult a qualified healthcare professional.",
        "hi": "⚠️ अस्वीकरण: मैं एक AI सहायक हूँ, डॉक्टर नहीं।\nकृपया किसी योग्य स्वास्थ्य विशेषज्ञ से सलाह लें।",
        "hi-en": "⚠️ Disclaimer: Main AI assistant hoon, doctor nahi.\nDoctor se zaroor consult karo."
    },
    "tip": {
        "en": "💡 Tip: Describe your symptoms and I will assess your cardiovascular risk.",
        "hi": "💡 सुझाव: अपने लक्षण बताएं और मैं आपके हृदय जोखिम का आकलन करूँगा।",
        "hi-en": "💡 Tip: Apne symptoms batao, main heart risk assess karunga."
    },
    "empty_input": {
        "en": "Please type or speak something so I can help you ❤️",
        "hi": "कृपया कुछ टाइप करें या बोलें ताकि मैं आपकी मदद कर सकूं ❤️",
        "hi-en": "Kuch type karo ya bolo taaki main help kar sakun ❤️"
    },
    "short_input": {
        "en": "Can you describe your symptoms in more detail?",
        "hi": "क्या आप अपने लक्षणों को और विस्तार से बता सकते हैं?",
        "hi-en": "Thoda aur detail mein symptoms batao."
    },
    "goodbye": {
        "en": "Stay heart healthy ❤️ Goodbye!",
        "hi": "हृदय स्वस्थ रखें ❤️ अलविदा!",
        "hi-en": "Heart healthy raho ❤️ Alvida!"
    }
}

# ===============================
# EXIT WORDS
# ===============================
EXIT_WORDS = {
    "en":   ["bye", "goodbye", "quit", "done", "exit"],
    "hi":   ["अलविदा", "बाय", "बंद करो", "छोड़ो"],
    "hi-en": ["bye", "alvida", "quit", "band karo"]
}

# ===============================
# LANGUAGE DETECTION SUPPORT
# ===============================
HINDI_CHARS = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")

HINGLISH_WORDS = [
    "mujhe", "hai", "ho", "kya", "nahi", "hoon", "tha", "thi",
    "dard", "sans", "thakaan", "seena", "dil", "bimar", "dawai",
    "doctor", "chest", "chalna", "chadna"
]

# Strong English indicators — if these dominate, it's English
ENGLISH_WORDS = [
    "the", "is", "are", "have", "has", "my", "i", "feel", "pain",
    "chest", "heart", "tired", "shortness", "breath", "symptom",
    "doctor", "health", "risk", "blood", "pressure", "dizzy"
]

# ===============================
# ✅ FIXED: LANGUAGE DETECTION (re-detects every message)
# ===============================
def detect_language(text: str, current_lang: str = None) -> str:
    """
    Detects language on EVERY message based on current input.
    Falls back to current_lang only if detection is uncertain.
    Returns: 'hi', 'hi-en', or 'en'
    """

    if not text or len(text.strip()) < 3:
        return current_lang or "en"

    # ✅ Step 1: Hindi script → definitely Hindi
    hindi_count = sum(1 for c in text if c in HINDI_CHARS)
    if hindi_count > 2:
        return "hi"

    lower = text.lower()
    words = lower.split()

    # ✅ Step 2: Count strong English word matches
    english_hits = sum(1 for w in words if w in ENGLISH_WORDS)
    if english_hits >= 2:
        return "en"

    # ✅ Step 3: langdetect for borderline cases
    try:
        detected = detect(text)
        if detected == "en":
            return "en"
        elif detected == "hi":
            # Could still be Hinglish (Roman script)
            hinglish_hits = sum(1 for w in HINGLISH_WORDS if w in lower)
            if hinglish_hits >= 2:
                return "hi-en"
            return "hi"
    except:
        pass

    # ✅ Step 4: Hinglish detection
    hinglish_hits = sum(1 for w in HINGLISH_WORDS if w in lower)
    if hinglish_hits >= 1:
        return "hi-en"

    # ✅ Step 5: Fall back to current session lang, then English
    return current_lang or "en"


# ===============================
# TRANSLATION HELPER
# ===============================
def t(key: str, lang: str) -> str:
    """Fetch translated text safely."""
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("en", key))


# ===============================
# PROMPT BUILDER (LLM)
# ===============================
def build_multilingual_prompt(user_input: str, risk_text: str, probability: float, lang: str) -> str:

    lang_instruction = {
        "en": "Respond ONLY in English. Do NOT use Hindi, Hinglish, or Devanagari script.",
        "hi": "केवल हिंदी में जवाब दें।",
        "hi-en": "Hinglish mein jawab do (Roman Hindi + English mix)."
    }.get(lang, "Respond in English.")

    return f"""
User said: {user_input}
Prediction: {risk_text}
Confidence: {probability:.2f}

You are a calm and helpful cardiovascular assistant.

{lang_instruction}

Guidelines:
- Keep response short (3-4 lines)
- Do not create panic
- If risk is high → suggest doctor
- Use simple human language
"""