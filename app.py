import streamlit as st
import numpy as np
from PIL import Image

from predictor import predict
from validate import validate_image

st.set_page_config(
    page_title="Facial Disease Detection",
    page_icon="🩺"
)

st.title("🩺 Facial Disease Detection")
st.write("Choose how you want to provide the image.")

# ==========================
# Choose Input
# ==========================
option = st.radio(
    "Choose Input",
    ["Upload Image", "Capture Image"]
)

image = None

if option == "Capture Image":
    image = st.camera_input("Capture Image")

elif option == "Upload Image":
    image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# ==========================
# Prediction
# ==========================
if image is not None:

    # Convert UploadedFile to PIL Image
    pil_image = Image.open(image)

    # Display image
    st.image(
        pil_image,
        caption="Input Image",
        use_container_width=True
    )

    # Convert PIL Image to NumPy
    img = np.array(pil_image)

    # Validate image
    valid, message = validate_image(img)

    if valid:

        st.success(message)

        disease, confidence, product, benefit = predict(img)

        st.success(f"Prediction: {disease}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Recommended Product")
        st.success(product)

        st.subheader("Benefits")
        st.info(benefit)

    else:

        st.error(message)
        st.stop()