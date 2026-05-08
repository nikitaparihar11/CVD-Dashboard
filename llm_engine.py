from groq import Groq
import re

client = Groq(api_key="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")

# ===============================
# LANGUAGE CONTROL
# ===============================
def get_lang_instruction(lang):
    if lang == "en":
        return (
            "You MUST respond in English ONLY. "
            "Do NOT use Hindi, Hinglish, Devanagari script, or any non-English words. "
            "If the user writes in Hindi, still reply in English only."
        )
    elif lang == "hi":
        return "केवल हिंदी में जवाब दें।"
    else:
        return "Hinglish mein jawab do — Hindi + English mix."

# ===============================
# QUESTIONS
# ===============================
QUESTIONS = {
    "en": [
        "How long have you had these symptoms?",
        "Does the pain spread to your arm, jaw, or back?",
        "Does physical activity make it worse?",
    ],
    "hi": [
        "यह लक्षण कितने समय से हैं?",
        "क्या दर्द कहीं और फैलता है?",
        "क्या चलने से तकलीफ बढ़ती है?",
    ],
    "hi-en": [
        "Symptoms kitne time se hain?",
        "Pain kahin spread hota hai?",
        "Activity se badhta hai?",
    ]
}

# ===============================
# RESPONSE FUNCTION
# ===============================
def generate_response(user_input, prediction, probability, history, lang="en"):
    lang_rule = get_lang_instruction(lang)
    user_turns = sum(1 for m in history if m["role"] == "user")

    import random
    question = random.choice(QUESTIONS.get(lang, QUESTIONS["en"]))

    # BEFORE DIAGNOSIS
    if user_turns < 4:
        task = f"Ask this follow-up question naturally: {question}"
    # AFTER DIAGNOSIS
    else:
        if prediction:
            verdict = {
                "en": "There may be signs of heart-related risk.",
                "hi": "दिल से जुड़ा जोखिम हो सकता है।",
                "hi-en": "Heart related risk ho sakta hai."
            }[lang]
            action = {
                "en": "Please consult a doctor as soon as possible.",
                "hi": "कृपया जल्द डॉक्टर से मिलें।",
                "hi-en": "Doctor se jaldi milo."
            }[lang]
        else:
            verdict = {
                "en": "No major heart risk is evident right now.",
                "hi": "अभी कोई बड़ा जोखिम नहीं दिख रहा।",
                "hi-en": "Abhi koi major risk nahi lag raha."
            }[lang]
            action = {
                "en": "Continue monitoring your symptoms.",
                "hi": "लक्षणों पर नज़र रखें।",
                "hi-en": "Symptoms monitor karo."
            }[lang]
        task = f"{verdict} {action}"

    # ===============================
    # BUILD PROMPT
    # ===============================
    messages = [{
        "role": "system",
        "content": (
            f"You are a cardiovascular disease (CVD) medical assistant.\n\n"
            f"LANGUAGE RULE (STRICT): {lang_rule}\n\n"
            f"RULES:\n"
            f"- Keep responses short: 1 to 3 sentences only\n"
            f"- Be warm and natural\n"
            f"- Never mix languages\n"
            f"- Never use Devanagari script when lang is English\n"
        )
    }]

    # ✅ FIX: Filter history properly per language
    for msg in history[-5:]:
        content = msg.get("content", "")
        if lang == "en":
            # Skip any message that contains Devanagari/Hindi characters
            if re.search(r'[\u0900-\u097F]', content):
                continue
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"{user_input}\n\nTask: {task}"
    })

    # ===============================
    # MODEL CALL
    # ===============================
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,   # ✅ Lower = more obedient to instructions
        max_tokens=120
    )

    reply = res.choices[0].message.content.strip()

    # ✅ FIX: Remove Devanagari characters specifically (not all non-ASCII)
    if lang == "en":
        reply = re.sub(r'[\u0900-\u097F]+', '', reply)  # Remove Devanagari only
        reply = re.sub(r'\s+', ' ', reply).strip()       # Clean up extra spaces

    return reply