import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from src.data import load_dataset
from src.utils import load_config

st.set_page_config(page_title="CCTV Classifier Dashboard", layout="centered")
st.title("📼 CCTV Object Classifier (CIFAR-10 / Fashion-MNIST)")

cfg = load_config("config.yaml")
(_, _), (_, _), _, input_shape, class_names = load_dataset(cfg["dataset"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(f"{cfg['models_dir']}/best_model.keras")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

    img_resized = img.resize((input_shape[1], input_shape[0]))
    arr = np.array(img_resized).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    arr = np.expand_dims(arr, 0)

    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    st.subheader(f"Prediction: **{class_names[top_idx]}** ({probs[top_idx]*100:.2f}%)")

    st.write("Class Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"- {class_names[i]}: {p*100:.2f}%")

st.info("Tip: Set dataset in config.yaml to 'cifar10' (matches cars/birds/etc.) or 'fashion_mnist' (28x28 grayscale).")
