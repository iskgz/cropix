import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# MODEL YÜKLE
model = tf.keras.models.load_model("fine_tuned_model.keras")

# DATASET
train_dir = Path("../dataset/train")

val_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128,128),
    batch_size=4,
)

# 🔥 ÖNCE LABEL AL (EN KRİTİK SATIR)
class_names = val_data.class_names

# PREPROCESS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
val_data = val_data.map(lambda x,y: (preprocess_input(x), y))

y_true = []
y_pred = []

for images, labels_batch in val_data:
    preds = model.predict(images)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()