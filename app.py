import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Page Config
st.set_page_config(page_title="Plant Care AI", page_icon="🍃")

# 2. CSS Fix: Using double braces {{ }} to prevent SyntaxErrors
st.markdown(f"""
<style>
.stApp {{
    background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
    background-attachment: fixed;
    background-size: cover;
}}

/* The Box that makes content visible */
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
    border: none;
}}

.result-card {{
    background-color: #e8f5e9;
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
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Error: '{model_path}' not found in the repository!")
        return None, None
        
    # Standard load
    model = load_model(model_path, compile=False)
    
    with open(label_path, "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_my_model()

# 4. App Interface
st.title("🪴 Plant Thirst Detector")
st.write("Upload a photo of your leaf to see if it's thirsty!")

if model is not None:
    choice = st.radio("Choose Input:", ["Camera", "Upload Image"], horizontal=True)

    if choice == "Camera":
        img_file = st.camera_input("Snap a photo")
    else:
        img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        # Preprocessing
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32)
        normalized_img = (img_array / 127.5) - 1
        data = np.expand_dims(normalized_img, axis=0)

        # Prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        st.markdown("---")
        if "Healthy" in class_name:
            st.balloons()
            st.markdown(f'<div class="result-card"><h3>✨ Result: {class_name[2:]}</h3><p>Confidence: {round(confidence * 100)}%</p><p>Your plant is happy! 🌸</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card" style="border-left-color: #ff8fa3;"><h3>💧 Result: {class_name[2:]}</h3><p>Confidence: {round(confidence * 100)}%</p><p>Time for a drink! 🚰</p></div>', unsafe_allow_html=True)
