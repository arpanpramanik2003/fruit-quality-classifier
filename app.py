import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Set page configuration for a wide layout and custom title
st.set_page_config(page_title="Fruit Quality Classifier", layout="wide", page_icon="🍎")

# Custom CSS for styling with a gradient background
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom, #e8f5e9, #fff3e0);
        padding: 20px;
    }
    .title {
        color: #2e7d32;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #31d816;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction {
        color: #d81b60;
        font-size: 20px;
        font-weight: bold;
    }
    .confidence {
        color: #1976d2;
        font-size: 18px;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #388e3c;
    }
    .stFileUploader, .stImage, .stCheckbox {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names (update based on your model)
class_names = ['Apple_Bad', 'Apple_Good', 'Banana_Bad', 'Banana_Good']

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get Grad-CAM heatmap
def direct_gradcam_heatmap(model, img_array, last_conv_layer_name='conv2d_35', pred_index=None):
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        layer_index = [layer.name for layer in model.layers].index(last_conv_layer_name)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            x = img_tensor
            conv_outputs = None
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if i == layer_index:
                    conv_outputs = x
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            raise ValueError("Gradients are None")
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), pred_index
    except Exception as e:
        st.error(f"Error in Grad-CAM: {e}")
        return None, None

# Streamlit interface
st.markdown('<div class="title">Fruit Quality Classifier 🍎🍌</div>', unsafe_allow_html=True)
st.markdown("Upload an image of an apple or banana to classify its quality and visualize model attention with Grad-CAM.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Preprocess and predict
    img_array = preprocess_image(image)
    if model is not None:
        with st.spinner("Classifying..."):
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = predictions[0][pred_index] * 100
            predicted_class = class_names[pred_index]
            
            st.markdown(f'<p class="prediction">Predicted Class: {predicted_class}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence">Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)
        
        # Grad-CAM option
        if st.checkbox("Show Grad-CAM Heatmap"):
            with st.spinner("Generating Grad-CAM..."):
                heatmap, _ = direct_gradcam_heatmap(model, img_array)
                if heatmap is not None:
                    # Overlay heatmap
                    img_np = np.array(image.resize((224, 224)))
                    heatmap = cv2.resize(heatmap, (224, 224))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = heatmap * 0.4 + img_np
                    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                    
                    st.markdown('<p class="subheader">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
                    st.image(superimposed_img, caption="Regions influencing the prediction", width=300)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow | Model trained on fruit quality dataset")