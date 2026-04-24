import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

class_names = [
    "wheat_healthy",
    "wheat_loosesmut",
    "wheat_powderymildew",
    "wheat_rust",
    "wheat_septoria"
]

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread("../dataset/test_images/wheat_loosesmut/cda_loose_smut_barley_alternate1.png")

if img is None:
    print("❌ test.jpg yok")
else:
    img = cv2.resize(img, (128,128))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    conf = np.max(output)

    print("Tahmin:", class_names[pred])
    print("Güven:", conf)