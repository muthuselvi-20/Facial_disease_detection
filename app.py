import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from predictor import predict
from validate import validate_image

st.title("🩺 Facial Disease Detection")

st.write("Capture your face using the webcam.")

image = st.camera_input("Capture Image")

if image is not None:

    # Convert UploadedFile → PIL
    pil_image = Image.open(image)

    # Convert PIL → NumPy
    img = np.array(pil_image)

    # Validate
    valid, message = validate_image(img)

    if valid:

        st.success(message)

        disease, confidence, product, benefit = predict(img)

        st.success(f"Prediction : {disease}")

        st.info(f"Confidence : {confidence:.2f}%")

        st.write("Product Recommendation: ")
        st.info(product)
        st.write("Benefits: ")
        st.info(benefit)

    else:

        st.error(message)

        st.stop()