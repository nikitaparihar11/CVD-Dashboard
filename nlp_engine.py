
# ===============================
# ADVANCED NLP ENGINE (Multilingual: EN + HI + Hinglish)
# ===============================
import nltk
from nltk.stem import WordNetLemmatizer
 
nltk.download('punkt',    quiet=True)
nltk.download('wordnet',  quiet=True)
nltk.download('punkt_tab',quiet=True)
 
lemmatizer = WordNetLemmatizer()
 
# ===============================
# SYMPTOM DEFINITIONS
# English + Hindi + Hinglish keywords
# ===============================
SYMPTOMS = {
    "chest_pain": {
        "keywords": [
            # English
            "chest pain", "pressure", "tightness", "burning", "chest discomfort",
            # Hindi (transliterated)
            "seene mein dard", "seene ka dard", "sina dard", "sina daba",
            "seena tight", "jalan", "dil mein dard",
            # Devanagari Hindi
            "सीने में दर्द", "सीने का दर्द", "सीना दर्द", "जलन", "दिल में दर्द"
        ],
        "weight": 2
    },
    "breath": {
        "keywords": [
            # English
            "shortness of breath", "breathless", "difficulty breathing", "cant breathe",
            # Hinglish
            "sans nahi aa rahi", "saans fulna", "saans lene mein takleef",
            "dam ghutna", "saans phoolna",
            # Devanagari
            "सांस नहीं आ रही", "सांस फूलना", "सांस लेने में तकलीफ", "दम घुटना"
        ],
        "weight": 2
    },
    "fatigue": {
        "keywords": [
            # English
            "tired", "fatigue", "weakness", "low energy", "exhausted", "lethargic",
            # Hinglish
            "thakaan", "kamzori", "energy nahi", "bahut thaka", "nira thak gaya",
            "thak gaya", "thak gayi", "kami",
            # Devanagari
            "थकान", "कमज़ोरी", "एनर्जी नहीं", "बहुत थका", "निराश"
        ],
        "weight": 1
    },
    "exercise": {
        "keywords": [
            # English
            "exercise", "walking", "running", "climbing stairs", "physical activity",
            # Hinglish
            "chalna", "daudna", "seedhi chadna", "vyayam", "exercise karna",
            # Devanagari
            "चलना", "दौड़ना", "सीढ़ी चढ़ना", "व्यायाम"
        ],
        "weight": 1
    }
}
 
# ===============================
# NEGATION WORDS (EN + HI + Hinglish)
# ===============================
NEGATIONS = [
    # English
    "no", "not", "never", "none", "without", "dont", "don't", "cannot", "can't",
    # Hinglish
    "nahi", "nahin", "mat", "nahi hai", "bilkul nahi",
    # Devanagari
    "नहीं", "मत", "बिल्कुल नहीं", "न"
]
 
# ===============================
# TEXT PREPROCESSING
# ===============================
def preprocess(text: str):
    text = text.lower()
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
 
# ===============================
# NEGATION CHECK
# ===============================
def is_negated(tokens, keyword_tokens, window=4):
    for i in range(len(tokens)):
        if tokens[i:i+len(keyword_tokens)] == keyword_tokens:
            start   = max(0, i - window)
            context = tokens[start:i]
            if any(neg in context for neg in NEGATIONS):
                return True
    return False
 
# ===============================
# MAIN NLP FUNCTION
# ===============================
def extract_symptoms(text: str) -> dict:
    tokens   = preprocess(text)
    text_str = " ".join(tokens)          # for multi-word matching
    raw_low  = text.lower()              # keep original for Devanagari
 
    detected = {key: 0 for key in SYMPTOMS}
 
    for symptom, info in SYMPTOMS.items():
        for keyword in info["keywords"]:
            kw_lower = keyword.lower()
            # Match in pre-processed tokens string OR in original lowercased text
            if kw_lower in text_str or kw_lower in raw_low:
                keyword_tokens = kw_lower.split()
                if is_negated(tokens, keyword_tokens):
                    detected[symptom] = 0
                else:
                    detected[symptom] = info["weight"]
                break   # one match per symptom is enough
 
    return detected
 
# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    tests = [
        "I have no fatigue but chest pain",
        "I have chest pain and feel very tired",
        "I don't have chest pain but I'm tired",
        "no shortness of breath and no weakness",
        # Hinglish tests
        "mujhe seene mein dard ho raha hai aur thakaan bhi hai",
        "saans nahi aa rahi aur bahut thaka hua hoon",
        "mujhe koi thakaan nahi hai",
        # Hindi tests
        "सीने में दर्द है और सांस फूल रही है",
        "थकान नहीं है लेकिन सीने में जलन है",
    ]
    for t in tests:
        print(f"\nInput : {t}")
        print(f"Output: {extract_symptoms(t)}")