import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.title("🐾 Cat vs Dog Classifier (ONNX version)")
st.write("Upload an image of a cat or dog to get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

class_names = ['Cat', 'Dog']

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    session = ort.InferenceSession("cat_dog_classifier.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})
    predicted_class = np.argmax(output[0])

    st.write(f"### 🧠 Prediction: **{class_names[predicted_class]}**")


