import tensorflow as tf

# model yükle
model = tf.keras.models.load_model("fine_tuned_model.keras")

# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 🔥 KAYDET (KESİN PATH)
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ model.tflite oluşturuldu")