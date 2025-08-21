import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from src.data import load_dataset
from src.utils import load_config

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="CCTV Classifier - Upload Image", layout="centered")
st.title("📷 CCTV Object Classifier")
st.write("Upload an image to classify using the trained CIFAR-10 or Fashion-MNIST model.")

# ========================
# LOAD CONFIG & DATA INFO
# ========================
cfg = load_config("config.yaml")
(_, _), (_, _), _, input_shape, class_names = load_dataset(cfg["dataset"])

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(f"{cfg['models_dir']}/best_model.keras")

model = load_model()

# ========================
# FILE UPLOADER
# ========================
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((input_shape[1], input_shape[0]))
    arr = np.array(img_resized).astype("float32") / 255.0
    if arr.ndim == 2:  # Grayscale for Fashion-MNIST
        arr = np.expand_dims(arr, -1)
    arr = np.expand_dims(arr, 0)

    # Prediction
    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    confidence = probs[top_idx] * 100

    # Display results
    st.subheader(f"Prediction: **{top_label}** ({confidence:.2f}%)")

    # Plot probabilities
    fig, ax = plt.subplots()
    ax.barh(class_names, probs * 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

st.info("Tip: Ensure 'dataset' in config.yaml is set to 'cifar10' or 'fashion_mnist' to match the model.")
