import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from pathlib import Path

# ================================
# 🔥 MODELİ YÜKLE
# ================================
model = load_model("best_model.h5")

# ================================
# 🔥 BASE MODELİ BUL
# ================================
base_model = None
for layer in model.layers:
    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

# ================================
# 🔥 KONTROL
# ================================
if base_model is None:
    raise Exception("Base model bulunamadı")

# ================================
# 🔥 FINE TUNE AÇ
# ================================
base_model.trainable = True

# ================================
# 🔥 SADECE SON KATMANLARI AÇ
# ================================
for layer in base_model.layers[:-10]:
    layer.trainable = False

# ================================
# 🔥 BatchNorm katmanlarını kapat (çok önemli)
# ================================
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# ================================
# 🔥 DÜŞÜK LEARNING RATE
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# 🔥 DATASET AYNI OLMALI
# ================================

train_dir = Path("../dataset/train")
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
SEED = 42

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# ================================
# 🔥 PREPROCESS (ÇOK KRİTİK)
# ================================
train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))

# ================================
# 🔥 CALLBACK
# ================================
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )
]

# ================================
# 🔥 FINE-TUNE EĞİTİM
# ================================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=callbacks
)

# ================================
# 🔥 KAYDET
# ================================
model.save("fine_tuned_model.keras")

print("Fine-tune tamam 🚀")