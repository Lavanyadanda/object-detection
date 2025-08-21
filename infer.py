import os, sys, numpy as np, tensorflow as tf
from PIL import Image
from src.data import load_dataset
from src.utils import load_config

def preprocess_image(img_path, input_shape):
    # Load and resize
    img = Image.open(img_path).convert("RGB")
    target_h, target_w = input_shape[0], input_shape[1]
    img = img.resize((target_w, target_h))
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)  # (1,H,W,C)

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/image.jpg")
        sys.exit(1)

    cfg = load_config("config.yaml")
    _, _, _, input_shape, class_names = load_dataset(cfg["dataset"])

    model_path = os.path.join(cfg["models_dir"], "best_model.keras")
    model = tf.keras.models.load_model(model_path)

    img_path = sys.argv[1]
    x = preprocess_image(img_path, input_shape)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    print(f"Predicted: {class_names[top_idx]} ({probs[top_idx]*100:.2f}%)")
    for i, p in enumerate(probs):
        print(f"{class_names[i]}: {p*100:.2f}%")

if __name__ == "__main__":
    main()
