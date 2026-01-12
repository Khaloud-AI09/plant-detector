import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Plant Care AI", page_icon="🍃")

# 2. Clean CSS (Using double braces {{ }} to avoid SyntaxErrors)
st.markdown(f"""
<style>
.stApp {{
    background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
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
}}
</style>
""", unsafe_allow_html=True)

# 3. Header
st.title("🪴 Plant Thirst Detector")
st.write("Upload a photo of your leaf or use the camera to see if it's thirsty!")

# 4. Load Model
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_my_model()

# 5. Input
choice = st.radio("Choose Input:", ["Camera", "Upload Image"], horizontal=True)

if choice == "Camera":
    img_file = st.camera_input("Snap a photo of your plant")
else:
    img_file = st.file_uploader("Upload a plant photo...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Checking this beauty...", use_container_width=True)
    
    # Process
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.markdown("---")
    if "Healthy" in class_name:
        st.balloons()
        st.markdown(f'<div class="result-card"><h3>✨ Result: {class_name[2:]}</h3><p>Confidence: {round(confidence_score * 100)}%</p><p>Your plant is thriving! 🌸</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-card" style="border-left-color: #ff8fa3;"><h3>💧 Result: {class_name[2:]}</h3><p>Confidence: {round(confidence_score * 100)}%</p><p>Time for a drink! 🚰</p></div>', unsafe_allow_html=True)