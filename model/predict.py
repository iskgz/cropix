import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore


DEFAULT_MODEL_PATH = Path("fine_tuned_model.keras")
DEFAULT_LABELS_PATH = Path("labels.json")
DEFAULT_IMAGE_PATH = Path("../dataset/test_images/wheat_rust/hububatta-kullenme-kok-bogazi-ve-sari-pas-alarmi.jpg")


def load_labels(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return {int(key): value for key, value in data.items()}


def load_image(path, image_size):
    image = tf.keras.utils.load_img(
        path,
        target_size=image_size,
        interpolation="bilinear",
    )
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0).astype(np.float32)
    return preprocess_input(array)


def split_label(label):
    parts = label.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "wheat", label


def predict_image(model, labels, image_path, top_k=3):
    image_size = tuple(model.input_shape[1:3])
    batch = load_image(image_path, image_size)
    probabilities = model.predict(batch, verbose=0)[0]

    top_ids = np.argsort(probabilities)[-top_k:][::-1]
    predictions = []
    for class_id in top_ids:
        label = labels.get(int(class_id), f"unknown_{int(class_id)}")
        plant, disease = split_label(label)
        predictions.append(
            {
                "class_id": int(class_id),
                "label": label,
                "plant": plant,
                "disease": disease,
                "confidence": float(probabilities[class_id]),
            }
        )

    return predictions


def predict(img_path, model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH):
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(labels_path)
    top_prediction = predict_image(model, labels, Path(img_path), top_k=1)[0]
    return (
        top_prediction["plant"],
        top_prediction["disease"],
        top_prediction["confidence"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Gorsel bulunamadi: {args.image}")

    model = tf.keras.models.load_model(args.model)
    labels = load_labels(args.labels)
    predictions = predict_image(model, labels, args.image, top_k=args.top_k)

    print(f"Gorsel: {args.image}")
    for index, item in enumerate(predictions, start=1):
        print(
            f"TOP{index}: {item['label']} "
            f"(%{item['confidence'] * 100:.2f})"
        )


if __name__ == "__main__":
    main()
