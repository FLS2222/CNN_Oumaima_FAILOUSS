{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e67e0d9-ceb7-4eba-a92f-60214de93c64",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from PIL import Image\n",
    "\n",
    "# Charger le modèle\n",
    "model = load_model(\"cat_dog_classifier.h5\")\n",
    "\n",
    "# Classes\n",
    "class_names = ['Cat', 'Dog']  # adapte selon l’ordre de tes classes\n",
    "\n",
    "st.title(\"🐾 Cat vs Dog Classifier\")\n",
    "st.write(\"Upload an image of a cat or dog to get the prediction.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    img = Image.open(uploaded_file).convert('RGB')\n",
    "    st.image(img, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    img = img.resize((128, 128))  # selon image_shape d'entraînement\n",
    "    img_array = image.img_to_array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "\n",
    "    prediction = model.predict(img_array)\n",
    "    predicted_class = np.argmax(prediction, axis=1)[0]\n",
    "\n",
    "    st.write(f\"### 🧠 Prediction: **{class_names[predicted_class]}**\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
