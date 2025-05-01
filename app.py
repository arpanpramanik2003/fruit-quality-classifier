import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Configuration ---
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Bad Apple', 'Good Apple', 'Bad Banana', 'Good Banana']

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# --- Preprocess Image ---
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Streamlit UI ---
st.title("🍎🍌 Fruit Quality Classification")

uploaded_file = st.file_uploader("Upload a fruit image (apple or banana)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")

    model = load_model()
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")
