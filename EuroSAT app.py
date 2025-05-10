import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
try:
    model = load_model("EuroSAT Model.h5")
except Exception as e:
    st.error(f"🚨 Failed to load model: {e}")
    st.stop()

# Define class names
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Define preprocessing function
def preprocess_image(img):
    img = img.resize((64, 64))  # Match training image size
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🌍 EuroSAT Land Use Classifier")
st.write("Upload a satellite image to classify its land use type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img_array = preprocess_image(image)
        st.write("✅ Image shape:", img_array.shape)
        st.write("🧠 Model expects:", model.input_shape)

        # Predict
        prediction = model.predict(img_array)
        st.write("📊 Prediction probabilities:", prediction)

        predicted_class = class_names[np.argmax(prediction)]
        st.markdown(f"### 🧠 Predicted Class: `{predicted_class}`")

    except Exception as e:
        st.error(f"⚠️ Something went wrong during prediction: {e}")
