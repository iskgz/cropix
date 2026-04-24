import joblib
import re
import numpy as np
from difflib import get_close_matches

# 🔥 LOAD
try:
    model = joblib.load("nlp_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model başarıyla yüklendi")
except Exception as e:
    print("❌ Model yüklenemedi:", e)
    model = None
    vectorizer = None

stopwords = ["yani", "da", "de", "ama"]

vocab = [
    "yaprak","yaprakta","basak","beyaz","toz","leke","sari",
    "kahverengi","siyah","benek","kabarcik","pudra",
    "yayiliyor","kuruma","sararma","kenar","orta","cerceve","un","unlu"
]

# 🔧 TYPO
def correct_typo(word):
    m = get_close_matches(word, vocab, n=1, cutoff=0.8)
    return m[0] if m else word

def typo_fix(text):
    return " ".join(correct_typo(w) for w in text.split())

# 🔧 PREPROCESS
def preprocess(text):
    text = str(text).lower().strip()

    text = (text.replace("ı","i").replace("ş","s").replace("ğ","g")
                .replace("ü","u").replace("ö","o").replace("ç","c"))

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    # Benzer kullanici ifadelerini ortak sinyallere indir.
    text = text.replace("una benzeyen", "un gibi")
    text = text.replace("unu andiran", "un gibi")
    text = text.replace("un serpilmis gibi", "un gibi")
    text = text.replace("un dokulmus gibi", "un gibi")
    text = text.replace("unlu", "un gibi")
    text = text.replace("kulleme", "beyaz toz")
    text = text.replace("kuf", "beyaz toz")
    text = text.replace("siyahlik", "siyah")
    text = text.replace("siyahliklar", "siyah")
    text = text.replace("kararmis", "kararma")
    text = text.replace("kararmislar", "kararma")
    text = text.replace("karariyor", "kararma")
    text = text.replace("kararmaya", "kararma")
    text = text.replace("kapkara", "siyah")
    text = text.replace("simsiyah", "siyah")
    text = text.replace("pudra","toz")
    text = text.replace("benek","leke")
    text = text.replace("kabarcik","leke")
    text = text.replace("un gibi", "beyaz toz")

    text = typo_fix(text)

    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)

# 🔥 NLP TAHMİN
def predict_nlp(text):
    if model is None:
        return "unknown", 0.0

    text = preprocess(text)

    if text == "":
        return "unknown", 0.0

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    label = model.classes_[np.argmax(probs)]
    confidence = float(np.max(probs))

    return label, confidence


def get_ranked_predictions(text, top_k=3):
    if model is None:
        return []

    clean_text = preprocess(text)
    if clean_text == "":
        return []

    X = vectorizer.transform([clean_text])
    probs = model.predict_proba(X)[0]
    sorted_idx = probs.argsort()[::-1][:top_k]

    return [(model.classes_[i], float(probs[i])) for i in sorted_idx]


def apply_answer_feedback(rankings, question_key, answer):
    if not rankings:
        return rankings

    score_map = {label: score for label, score in rankings}
    answer = answer.lower().strip()

    if question_key == "toz":
        if "hayir" in answer:
            score_map["septoria"] = score_map.get("septoria", 0.0) + 0.24
            score_map["rust"] = score_map.get("rust", 0.0) - 0.22
            score_map["powderymildew"] = score_map.get("powderymildew", 0.0) - 0.28
        else:
            score_map["rust"] = score_map.get("rust", 0.0) + 0.18
            score_map["powderymildew"] = score_map.get("powderymildew", 0.0) + 0.16
            score_map["septoria"] = score_map.get("septoria", 0.0) - 0.18

    if question_key == "renk":
        if "beyaz" in answer:
            score_map["powderymildew"] = score_map.get("powderymildew", 0.0) + 0.20
            score_map["rust"] = score_map.get("rust", 0.0) - 0.18
        elif "sari" in answer:
            score_map["rust"] = score_map.get("rust", 0.0) + 0.20
            score_map["powderymildew"] = score_map.get("powderymildew", 0.0) - 0.18
            score_map["septoria"] = score_map.get("septoria", 0.0) + 0.06

    if question_key == "leke_merkez":
        if "evet" in answer:
            score_map["septoria"] = score_map.get("septoria", 0.0) + 0.18
            score_map["rust"] = score_map.get("rust", 0.0) - 0.05
        elif "hayir" in answer:
            score_map["rust"] = score_map.get("rust", 0.0) + 0.08
            score_map["septoria"] = score_map.get("septoria", 0.0) - 0.06

    if question_key == "yayilim":
        if "evet" in answer:
            score_map["powderymildew"] = score_map.get("powderymildew", 0.0) + 0.08
            score_map["rust"] = score_map.get("rust", 0.0) + 0.05
        elif "hayir" in answer:
            score_map["septoria"] = score_map.get("septoria", 0.0) + 0.08

    adjusted = [(label, max(0.0, score)) for label, score in score_map.items()]
    adjusted.sort(key=lambda x: x[1], reverse=True)

    total = sum(score for _, score in adjusted)
    if total > 0:
        adjusted = [(label, score / total) for label, score in adjusted]

    return adjusted

# 🔥 SORU ÜRET
def generate_question(top1, top2, text, asked_questions):
    text = text.lower()

    candidates = []

    if {"rust", "septoria"} == {top1, top2}:
        candidates.append(("leke_merkez", "Lekelerin ortası daha açık renkli mi? (evet/hayır)"))
        candidates.append(("toz", "Toz var mı? (evet/hayır)"))

    if {"powderymildew", "rust"} == {top1, top2}:
        candidates.append(("renk", "Renk beyaz mı sarı mı?"))
        candidates.append(("toz", "Toz veya pudra gibi bir yapı var mı? (evet/hayır)"))

    if "loosesmut" in [top1, top2]:
        candidates.append(("basak", "Başakta mı görülüyor? (evet/hayır)"))

    # Genel soru en sona kalsin; once daha ayirt edici sorular sorulsun.
    candidates.append(("yayilim", "Belirti yayılıyor mu? (evet/hayır)"))

    for question_key, question_text in candidates:
        if question_key in asked_questions:
            continue

        if question_key == "toz" and ("toz" in text or "pudra" in text):
            continue
        if question_key == "renk" and ("beyaz" in text or "sari" in text):
            continue
        if question_key == "basak" and "basak" in text:
            continue
        if question_key == "yayilim" and "yayil" in text:
            continue

        return question_key, question_text

    return None, None

# 🔥 CEVAP NORMALİZASYON
def normalize_answer(question, answer):
    answer = answer.lower()

    if "toz" in question:
        return "toz yok" if "hayir" in answer else "toz var"

    if "beyaz" in question or "sari" in question:
        return answer

    if "basak" in question:
        return "basak" if "evet" in answer else "yaprak"

    if "orta" in question or "acik renkli" in question:
        return "orta acik" if "evet" in answer else "orta koyu"

    if "yayiliyor" in question or "yayılıyor" in question:
        return "yayiliyor" if "evet" in answer else "yayilmiyor"

    return answer

# 🔥 ANA SİSTEM
def interactive_predict():
    text = input("\nBelirtiyi yaz (q çık): ")

    if text == "q":
        return False

    max_round = 4
    asked_questions = set()
    feedback_rankings = []

    for i in range(max_round):
        rankings = feedback_rankings if feedback_rankings else get_ranked_predictions(text, top_k=4)
        if not rankings:
            print("👉 Tahmin: unknown | Güven: 0.00")
            return True

        label, conf = rankings[0]
        print(f"👉 Tahmin: {label} | Güven: {conf:.2f}")

        # 🔥 yeterli güven
        if conf >= 0.75:
            print("✅ Teşhis tamamlandı")
            return True

        if len(rankings) < 2:
            print("❌ Daha fazla soru üretilemedi")
            return True

        top1 = rankings[0][0]
        top2 = rankings[1][0]
        question_key, question = generate_question(top1, top2, text, asked_questions)

        if not question:
            print("✅ Ek soru gerekmiyor, mevcut en güçlü tahmin korundu")
            return True

        answer = input("💬 " + question + " ")
        asked_questions.add(question_key)

        # 🔥 EN KRİTİK
        text += " " + normalize_answer(question, answer)
        feedback_rankings = apply_answer_feedback(rankings, question_key, answer)

        if feedback_rankings:
            new_label, new_conf = feedback_rankings[0]
            second_conf = feedback_rankings[1][1] if len(feedback_rankings) > 1 else 0.0

            if new_conf >= 0.72 and (new_conf - second_conf) >= 0.12:
                print(f"👉 Güncellenmiş tahmin: {new_label} | Güven: {new_conf:.2f}")
                print("✅ Sorularla teşhis tamamlandı")
                return True

    print("⚠️ Maksimum soru sayısına ulaşıldı")
    return True

# 🔥 DIREKT BAŞLA
if __name__ == "__main__":
    while True:
        cont = interactive_predict()
        if cont is False:
            break
