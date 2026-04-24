from nlp.predict_nlp import predict_nlp, preprocess

def ask_questions(text):
    questions = []

    text = text.lower()

    if "siyah" not in text:
        questions.append("Yaprakta siyah noktalar var mı?")

    if "alt yaprak" not in text:
        questions.append("Hastalık alt yapraklardan mı başladı?")

    if "gri" not in text and "kahverengi" not in text:
        questions.append("Lekelerin ortası açık renkli mi?")

    if "toz" not in text and "beyaz" not in text:
        questions.append("Yaprakta toz veya pudra gibi bir yapı var mı?")

    return questions


def interactive_nlp():

    user_text = input("Belirtileri yaz: ")

    # 🔥 preprocess ekledik (KRİTİK)
    clean_text = preprocess(user_text)

    pred, conf = predict_nlp(clean_text)

    print("👉 İlk tahmin:", pred, "| Güven:", conf)

    questions = ask_questions(user_text)

    for q in questions:
        answer = input(q + " ")

        # 🔥 cevapları da preprocess ile ekle
        user_text += " " + answer

    # 🔥 tekrar temizle
    clean_text = preprocess(user_text)

    pred2, conf2 = predict_nlp(clean_text)

    print("👉 Güncellenmiş tahmin:", pred2, "| Güven:", conf2)

    return pred2, conf2