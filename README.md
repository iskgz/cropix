# Cropix - Bugday Hastalik Tespit Projesi (Goruntu Isleme + NLP)

Bu proje, bugday bitkisindeki hastalik belirtilerini iki farkli yaklasimla analiz eder.

- Goruntu isleme (CNN / MobileNetV2): Yaprak veya basak goruntusunden hastalik tahmini yapar.
- NLP (metin analizi): Kullanicinin yazdigi belirti aciklamasindan hastalik tahmini yapar.

Proje, hem goruntu tabanli hem metin tabanli tani adimlarini ayni calismada birlestirmeyi hedefler.

## Proje Amaci

Bu calismanin amaci:

- Bugday hastaliklarini otomatik tespit etmek
- Farkli veri tiplerini (gorsel + metin) birlikte kullanmak
- Hastalik tespit surecini hizlandirmak ve karar destegi saglamak

## Kapsanan Hastalik Siniflari

Model dosyalarinda gecen temel siniflar:

- wheat_healthy
- wheat_loosesmut
- wheat_powderymildew
- wheat_rust
- wheat_septoria

## Proje Dizini

```text
cropix/
├── dataset/
├── model/
│   ├── train.py
│   ├── predict.py
│   ├── test_tflite.py
│   ├── labels.json
│   ├── model.tflite
│   └── (egitilmis .h5/.keras dosyalari)
├── nlp/
│   ├── train_nlp.py
│   ├── predict_nlp.py
│   ├── interactive_nlp.py
│   ├── big_dataset.csv
│   └── (kaydedilmis .pkl modeller)
├── main.py
└── README.md
```

## Kullanilan Yontemler

### 1) Goruntu Isleme Tarafi (CNN)

- `model/train.py`
  - `dataset/train` altindaki goruntulerden egitim yapar
  - Transfer learning olarak `MobileNetV2` kullanir
  - Dengesiz siniflar icin `class_weight` uygular
  - En iyi modeli kaydetme, erken durdurma ve ogrenme hizi azaltma callbackleri vardir

- `model/predict.py`
  - Egitilmis modeli yukler
  - Verilen bir gorsel icin en yuksek olasilikli siniflari (`top-k`) verir

- `model/test_tflite.py`
  - `model.tflite` dosyasi ile TFLite inferans testi yapar

### 2) NLP Tarafi

- `nlp/train_nlp.py`
  - `big_dataset.csv` verisini temizler ve on isleme uygular
  - TF-IDF (`ngram_range=(1,3)`) + Logistic Regression modeli egitir
  - `nlp_model.pkl` ve `vectorizer.pkl` dosyalarini kaydeder

- `nlp/predict_nlp.py`
  - Kayitli NLP modelini yukler
  - Kullanicidan gelen metni normalize edip sinif ve guven skoru uretir
  - Gerekirse ayirt edici sorularla puanlari guncelleyebilecek yardimci fonksiyonlar icerir

- `nlp/interactive_nlp.py`
  - Kullaniciya soru-cevap akisiyla ek belirti bilgisi toplar
  - Ilk tahmini ve guncellenmis tahmini ekrana yazar

### 3) Birlesik Akis

- `main.py`
  - Once goruntu tahmini alir
  - Sonra gerekirse NLP adimina gecer
  - Iki farkli guven degerini agirlikli birlestirerek final karar mantigi uygular

## Kurulum

Asagidaki adimlar Ubuntu/WSL terminali icindir.

1. Proje klasorune gec:

```bash
cd /home/projeGÜNCEL/cropix
```

2. (Opsiyonel ama onerilir) Sanal ortam olustur:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Gerekli paketleri yukle:

```bash
pip install --upgrade pip
pip install tensorflow numpy pandas scikit-learn joblib opencv-python
```

## Calistirma Adimlari

### A) NLP Modeli Egitmek

```bash
cd /home/projeGÜNCEL/cropix/nlp
python3 train_nlp.py
```

Beklenen cikti:

- Siniflandirma raporu (`classification_report`)
- `nlp_model.pkl` ve `vectorizer.pkl` dosyalari

### B) NLP Tahmin / Etkilesimli Test

```bash
cd /home/projeGÜNCEL/cropix/nlp
python3 interactive_nlp.py
```

### C) Goruntu Modeli Egitmek

```bash
cd /home/eda/projeGÜNCEL/cropix/model
python3 train.py
```

### D) Goruntuden Tahmin Almak

```bash
cd /home/projeGÜNCEL/cropix/model
python3 predict.py ../dataset/test_images/wheat_rust/hububatta-kullenme-kok-bogazi-ve-sari-pas-alarmi.jpg
```

### E) TFLite Testi

```bash
cd /home/eda/projeGÜNCEL/cropix/model
python3 test_tflite.py
```

## Veri Hakkinda Not

- `dataset/` klasoru goruntu verisini barindirir
- `nlp/big_dataset.csv` metin tabanli belirti verilerini icerir
- Buyuk model dosyalari (`.h5`, `.keras`, `.tflite`, `.pkl`) depoda bulunmaktadir

## Gelistirme Onerileri

- `requirements.txt` eklenmesi
- Egitim/deney sonuclarinin tek raporda toplanmasi (accuracy, f1, confusion matrix)
- `main.py` icin daha net bir demo akisi ve ornek girdi dosyasi tanimi
- Veri seti surumleme ve model versiyonlama

## GitHub

Bu proje GitHub uzerinden paylasilmaktadir:

- Repo: https://github.com/iskgz/cropix
