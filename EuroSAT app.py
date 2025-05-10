import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("EuroSAT Model.h5")

# Define classes
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Define image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # normalize
    if img_array.shape[-1] == 4:  # remove alpha if exists
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

# Streamlit app
st.title("🌍 EuroSAT Land Use Classifier")
st.write("Upload a satellite image to classify its land use type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.markdown(f"### 🧠 Predicted Class: {predicted_class}")
