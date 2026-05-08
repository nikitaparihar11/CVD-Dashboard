# ===============================
# CVD CHATBOT — Main Entry Point
# ===============================

import sys
sys.stdout.reconfigure(encoding='utf-8')   # fix Hindi display in terminal

import joblib
import pandas as pd

from nlp_engine     import extract_symptoms
from llm_engine     import generate_response
from who_giudelines import load_pdf_chunks, create_index, search
from lang_engine    import t, select_language, detect_language, EXIT_WORDS
from voice_engine   import listen, VOICE_AVAILABLE

# ===============================
# LOAD MODEL & FEATURES
# ===============================
model    = joblib.load("heart_model.pkl")
features = joblib.load("features.pkl")

# ===============================
# LOAD PDF + CREATE INDEX
# ===============================
chunks            = load_pdf_chunks("who_guidelines.pdf")
index, embeddings = create_index(chunks)

# ===============================
# CONVERT NLP → MODEL FEATURES
# ===============================
def symptoms_to_features(symptoms: dict) -> dict:
    return {
        "age":      50,
        "sex":       1,
        "trestbps": 120,
        "chol":     200,
        "fbs":        0,
        "restecg":    1,
        "thalach":  150,
        "exang":    1 if symptoms.get("exercise", 0) > 0 else 0,
        "oldpeak":  1.5 if symptoms.get("chest_pain", 0) > 0 else 0.0,
        "ca":         0,
        "cp":        1 if symptoms.get("chest_pain", 0) > 0 else 0,
    }

# ===============================
# PREPARE MODEL INPUT
# ===============================
def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    return df

# ===============================
# CONVERSATION HISTORY
# ===============================
conversation_history = []

# ===============================
# LANGUAGE SELECTION (startup default)
# ===============================
lang = select_language()

# ===============================
# WELCOME MESSAGE
# ===============================
print("=" * 50)
print(f"   {t('welcome_title', lang)}")
print("=" * 50)
print(f"\n{t('disclaimer', lang)}\n")
print(t('tip', lang))
if VOICE_AVAILABLE:
    print("🎤 Voice input: type 'v' and press ENTER to speak")
print("=" * 50 + "\n")

# ===============================
# CHATBOT LOOP
# ===============================
while True:

    # --------------------
    # INPUT
    # --------------------
    if VOICE_AVAILABLE:
        raw = input("You (or 'v' for voice): ").strip()
        if raw.lower() == "v":
            user_input = listen(duration=7)
            if not user_input:
                print("Bot: Could not understand. Please type your symptoms.\n")
                continue
            print(f"✅ Heard: {user_input}\n")
        else:
            user_input = raw
    else:
        user_input = input("You: ").strip()

    # --------------------
    # ✅ FIX: Auto-detect language per message, update session lang
    # --------------------
    lang        = detect_language(user_input, current_lang=lang)
    active_lang = lang

    # --------------------
    # INPUT VALIDATION
    # --------------------
    if not user_input:
        print(f"Bot: {t('empty_input', active_lang)}\n")
        continue

    if len(user_input) < 3:
        print(f"Bot: {t('short_input', active_lang)}\n")
        continue

    # Check exit words across all languages
    lower_input = user_input.lower().strip()
    all_exits = EXIT_WORDS.get("en", []) + EXIT_WORDS.get(active_lang, [])
    if lower_input in all_exits:
        print(f"Bot: {t('goodbye', active_lang)}")
        break

    # --------------------
    # STEP 1: NLP — extract symptoms
    # --------------------
    symptoms = extract_symptoms(user_input)

    # --------------------
    # STEP 2: Build model input
    # --------------------
    data = symptoms_to_features(symptoms)
    df   = prepare_input(data)

    # --------------------
    # STEP 3: Prediction
    # --------------------
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # --------------------
    # STEP 4: WHO Guidelines search
    # --------------------
    context = search(user_input, chunks, index)

    # --------------------
    # STEP 5: Add user message to history
    # --------------------
    conversation_history.append({"role": "user", "content": user_input})

    # --------------------
    # STEP 6: Generate LLM response
    # --------------------
    reply = generate_response(
        user_input + "\n\nWHO Guidelines Context:\n" + context,
        pred,
        prob,
        history=conversation_history,
        lang=active_lang
    )

    # --------------------
    # STEP 7: Save reply & display
    # --------------------
    conversation_history.append({"role": "assistant", "content": reply})  # ✅ FIXED: "assistant" not "bot"
    print(f"\nBot: {reply}\n")
    print("-" * 50)