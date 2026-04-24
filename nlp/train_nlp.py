import pandas as pd
import joblib
import re
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 📁 DATASET
df = pd.read_csv("big_dataset.csv", on_bad_lines="skip")

# 🔥 STOPWORDS
stopwords = ["yani", "da", "de", "ama"]

# 🔥 VOCAB
vocab = [
    "yaprak", "yaprakta", "basak", "beyaz", "toz", "leke", "sari",
    "kahverengi", "siyah", "benek", "kabarcik", "pudra",
    "yayiliyor", "kuruma", "sararma", "kenar", "orta", "un", "unlu"
]

# 🔥 TYPO FIX
def correct_typo(word):
    matches = get_close_matches(word, vocab, n=1, cutoff=0.8)
    return matches[0] if matches else word

def typo_fix(text):
    return " ".join(correct_typo(w) for w in text.split())

# 🔥 PREPROCESS
def preprocess(text):
    text = str(text).lower().strip()

    text = text.replace("ı", "i").replace("ş", "s").replace("ğ", "g") \
               .replace("ü", "u").replace("ö", "o").replace("ç", "c")

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
    text = text.replace("pudra", "toz")
    text = text.replace("toz gibi", "toz")
    text = text.replace("benek", "leke")
    text = text.replace("kabarcik", "leke")
    text = text.replace("un gibi", "beyaz toz")

    text = typo_fix(text)

    words = text.split()
    words = [w for w in words if w not in stopwords]

    return " ".join(words)

# 🔥 TEMİZLE
# Once gercek bos degerleri sil ki "nan" string sinifina donusmesinler.
df = df.dropna(subset=["text", "label"])

df["text"] = df["text"].apply(preprocess)
df["label"] = df["label"].astype(str).str.lower().str.strip()

df = df[df["text"] != ""]
df = df[df["label"] != ""]
df = df[df["label"] != "nan"]
df = df.drop_duplicates()

# 🔥 DENGELEME
df = df.groupby("label").apply(lambda x: x.sample(min(len(x), 300))).reset_index(drop=True)

# 🔥 SPLIT
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 🔥 TF-IDF (UPGRADE)
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=8000,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 MODEL (EN KRİTİK DEĞİŞİM)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# 🔥 TEST
y_pred = model.predict(X_test_vec)

print("\n=== RAPOR ===")
print(classification_report(y_test, y_pred))

print("\nSınıflar:", model.classes_)

# 🔥 KAYDET
joblib.dump(model, "nlp_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n🔥 Model kaydedildi!")
