import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from pathlib import Path
import gdown

# 📥 Télécharger le modèle depuis Google Drive si nécessaire
model_path = "cat_dog_classifier.h5"
if not Path(model_path).exists():
    file_id = "1yNL7jzVQlKtLfQQ8WEfR0XjhM9I70XWV"
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info("Téléchargement du modèle...")
    gdown.download(url, model_path, quiet=False)

# 🚀 Charger le modèle
model = load_model(model_path)

# Noms des classes (doivent correspondre à ton entraînement)
class_names = ['Cat', 'Dog']

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog to get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.success(f"### 🧠 Prediction: **{class_names[predicted_class]}**")

