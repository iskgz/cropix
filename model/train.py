# ================================
# 📦 KÜTÜPHANELER
# ================================

import json                          # Label kayıt etmek için
import os                            # Sistem ayarları
from pathlib import Path             # Dosya yolları
from collections import Counter      # Sınıf sayımı

import tensorflow as tf              # Ana deep learning kütüphanesi
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore


# ================================
# ⚙️ SİSTEM AYARLARI (RAM / CPU KONTROL)
# ================================

# CPU thread sınırlandırma (RAM patlamasını azaltır)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

# XLA kapat (stabilite için)
tf.config.optimizer.set_jit(False)

# Random seed (tekrarlanabilir sonuç)
tf.random.set_seed(42)


# ================================
# 🎮 GPU KONTROL
# ================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        # GPU belleğini anlık kullan (bir anda doldurmaz)
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU aktif")
else:
    print("⚠️ GPU bulunamadı, CPU ile çalışacak")


# ================================
# 📂 VERİ YOLU VE PARAMETRELER
# ================================

train_dir = Path("../dataset/train")   # Dataset yolu

IMG_SIZE = (128, 128)   # 🔥 RAM dostu (224 yerine)
BATCH_SIZE = 4          # 🔥 Küçük batch (RAM kontrol)
SEED = 42               # Sabit seed


# ================================
# 📊 SINIF SAYIMI (CLASS WEIGHT İÇİN)
# ================================

def count_images_by_class(root):
    """
    Dataset içindeki her klasörde kaç görüntü olduğunu sayar.
    Bu sayede dengesiz veri problemini çözmek için class_weight hesaplanır.
    """
    counts = Counter()

    for class_dir in root.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*")))

    return counts


class_counts = count_images_by_class(train_dir)


# ================================
# 🧠 DATASET YÜKLEME
# ================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,   # %80 train
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,   # %20 validation
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_data.class_names
num_classes = len(class_names)


# ================================
# ⚖️ CLASS WEIGHT (DENGESİZ VERİ ÇÖZÜMÜ)
# ================================

max_count = max(class_counts.values())

class_weight = {
    class_names.index(name): (max_count / count) ** 0.5
    for name, count in class_counts.items()
}


# ================================
# 🔄 DATA AUGMENTATION (HAFİF)
# ================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # Yatay çevirme
    layers.RandomRotation(0.05),       # Hafif döndürme
])


# ================================
# 🔧 VERİ İŞLEME PIPELINE
# ================================

def process(x, y):
    """
    Tek pipeline içinde:
    - Augmentation
    - Normalizasyon
    """
    x = data_augmentation(x, training=True)
    x = preprocess_input(x)
    return x, y


# Train dataset işlemleri
train_data = train_data.map(
    process,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Validation dataset işlemleri (augmentation yok)
val_data = val_data.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Prefetch ile veri akışı hızlandırılır
train_data = train_data.prefetch(tf.data.AUTOTUNE)
val_data = val_data.prefetch(tf.data.AUTOTUNE)


# ================================
# 🧠 MODEL OLUŞTURMA (TRANSFER LEARNING)
# ================================

base_model = MobileNetV2(
    weights="imagenet",      # Önceden eğitilmiş model
    include_top=False,       # Son katman yok
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    alpha=0.75               # 🔥 Model küçültme
)

# İlk aşamada base model dondurulur
base_model.trainable = False


# ================================
# 🔗 MODEL MİMARİSİ
# ================================

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    64,   # 🔥 Küçük dense layer
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4)
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    dtype="float32"
)(x)

model = models.Model(inputs, outputs)


# ================================
# 📉 CALLBACKS (EĞİTİM KONTROLÜ)
# ================================

callbacks = [

    # En iyi modeli kaydeder
    ModelCheckpoint(
        "best_model.h5",
        save_best_only=True,
        monitor="val_accuracy"
    ),

    # Öğrenme hızını düşürür
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2
    ),

    # Erken durdurma (overfitting önleme)
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )
]


# ================================
# ⚙️ MODEL DERLEME
# ================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ================================
# 🚀 MODEL EĞİTİMİ
# ================================

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks,
)


# ================================
# 💾 MODEL KAYDETME
# ================================

model.save("model.h5")

# Label mapping kaydet
labels = {i: name for i, name in enumerate(class_names)}

with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)


print("✅ Eğitim tamamlandı!")