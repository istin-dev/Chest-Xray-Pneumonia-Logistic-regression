import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
# Correct path handling
base_path = os.path.dirname(__file__)

# Load models (use correct filenames)
model = joblib.load(os.path.join(base_path, "pca_model.pkl"))
pca = joblib.load(os.path.join(base_path, "pneumonia_logistic_model_final_pca_30.pkl"))

st.title("Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image to detect Pneumonia")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocessing (same as training)
    image = image.convert('L')
    image = image.resize((100,100))

    img_array = np.array(image)
    img_array = img_array.flatten().reshape(1,-1)

    # PCA transform
    img_pca = pca.transform(img_array)

    # Prediction
    prediction = model.predict(img_pca)

    # Probability
    prob = model.predict_proba(img_pca)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Pneumonia Detected")
    else:
        st.success("Normal Lung")

    st.write("Confidence:", round(np.max(prob)*100,2),"%")
