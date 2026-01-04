import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 1. Page Config & Aesthetic CSS
st.set_page_config(page_title="Plant Care AI", page_icon="🍃")

st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #C5D89D;
    }
    h1 {
        color: #043915;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #BBC863;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
    }
    /* Cute container for results */
    .result-card {
        background-color: #8BAE66;
        padding: 20px;
        border-radius: 15px;
        border-left: 10px solid #74c69d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar / Header
st.title("🪴 Plant Thirst Detector")
st.write("Upload a photo of your leaf or use the camera to see if it's thirsty!")

# 3. Load the Model
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_my_model()

# 4. Image Input (Aesthetic Choice: Upload or Camera)
choice = st.radio("Choose Input:", ["Camera", "Upload Image"], horizontal=True)

if choice == "Camera":
    img_file = st.camera_input("Snap a photo of your plant")
else:
    img_file = st.file_uploader("Upload a plant photo...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Prepare the image
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Checking this beauty...", use_container_width=True)
    
    # Process image for the model
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
    confidence_score = prediction[0][index]

    # 5. Aesthetic Result Display
    st.markdown("---")
    if "Healthy" in class_name:
        st.balloons()
        st.markdown(f"""
            <div class="result-card">
                <h3>✨ Result: {class_name[2:]}</h3>
                <p>Confidence: {round(confidence_score * 100)}%</p>
                <p>Your plant is thriving! Keep doing what you're doing. 🌸</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card" style="border-left-color: #ff8fa3;">
                <h3>💧 Result: {class_name[2:]}</h3>
                <p>Confidence: {round(confidence_score * 100)}%</p>
                <p>Oh no! Your plant looks a bit thirsty. Time for a drink! 🚰</p>
            </div>

        """, unsafe_allow_html=True)
