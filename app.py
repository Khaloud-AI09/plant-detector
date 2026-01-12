import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Page Config
st.set_page_config(page_title="Plant Care AI", page_icon="🍃")

# 2. CSS with double curly braces to avoid SyntaxErrors
st.markdown(f"""
<style>
.stApp {{
    background-image: url("https://images.unsplash.com/photo-1470058869958-2a77ade41c02?q=80&w=870&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-attachment: fixed;
    background-size: cover;
}}
.block-container {{
    background-color: rgba(255, 255, 255, 0.9);
    padding: 3rem !important;
    border-radius: 20px;
    margin-top: 2rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}}
h1, h2, h3, p, span, label {{
    color: #043915 !important;
}}
.stButton>button {{
    background-color: #BBC863;
    color: white !important;
    border-radius: 20px;
    width: 100%;
}}
.result-card {{
    background-color: #8BAE66;
    padding: 20px;
    border-radius: 15px;
    border-left: 10px solid #74c69d;
    color: #043915;
}}
</style>
""", unsafe_allow_html=True)

# 3. Robust Model Loading
@st.cache_resource
def load_my_model():
    model_path = "keras_model.h5"
    label_path = "labels.txt"
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in repository!")
        return None, None
        
    # Standard load for Keras H5 files
    model = load_model(model_path, compile=False)
    
    with open(label_path, "r") as f:
        class_names = f.readlines()
    return model, class_names

# Initialize
model, class_names = load_my_model()

# 4. App UI
st.title("🪴 Plant Thirst Detector")

if model is not None:
    choice = st.radio("Choose Input:", ["Camera", "Upload Image"], horizontal=True)

    if choice == "Camera":
        img_file = st.camera_input("Snap a photo of your plant")
    else:
        img_file = st.file_uploader("Upload a plant photo...", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Analyzing your plant...", use_container_width=True)
        
        # Preprocessing
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        st.markdown("---")
        if "Healthy" in class_name:
            st.balloons()
            st.markdown(f'<div class="result-card"><h3>✨ Result: {class_name[2:]}</h3><p>Confidence: {round(confidence * 100)}%</p><p>Your plant is doing great! Keep it up. 🌸</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card" style="border-left-color: #ff8fa3;"><h3>💧 Result: {class_name[2:]}</h3><p>Confidence: {round(confidence * 100)}%</p><p>It looks a bit thirsty. Time for a drink! 🚰</p></div>', unsafe_allow_html=True)
else:
    st.warning("Please upload 'keras_model.h5' and 'labels.txt' to your GitHub repository.")
