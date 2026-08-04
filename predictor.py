import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input



model = load_model(r"D:\Projects\Facial problem detection\Model\facial_model.keras")

class_names = [
    "Acne",
    "Eczemaa",
    "Hyperpigmentation",
    "Normal",
    "Rosacea",
    "Wrinkle"
]

product_recommendations = {
    "Acne": {
        "product": "Salicylic Acid (2%) Cleanser",
        "benefit": "BHA exfoliates inside pores, cuts oil and breakouts."
    },

    "Eczemaa": {
        "product": "Ceramide-based Moisturizing Cream",
        "benefit": "Repairs the skin barrier, reduces flare-ups and dryness."
    },

    "Hyperpigmentation": {
        "product": "Vitamin C Serum (10–15%)",
        "benefit": "Brightens dark spots and helps reduce melanin production."
    },

    "Normal": {
        "product": "Lightweight Hyaluronic Acid Moisturizer",
        "benefit": "Maintains skin hydration without disrupting the natural balance."
    },

    "Rosacea": {
        "product": "Azelaic Acid Cream (10%)",
        "benefit": "Reduces redness and inflammation while being gentle on sensitive skin."
    },

    "Wrinkle": {
        "product": "Retinol Serum (0.3–0.5%)",
        "benefit": "Boosts collagen production and reduces fine lines over time."
    }
}

def preprocess_image(image):
    """
    image -> NumPy array (RGB)
    """

    # Convert NumPy array to PIL Image
    img = Image.fromarray(image)

    # Convert to RGB
    img = img.convert("RGB")

    # Resize
    img = img.resize((224, 224))

    # Convert back to NumPy
    img = np.array(img).astype("float32")

    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index]) * 100

    disease = class_names[index]

    product = product_recommendations[disease]["product"]
    benefit = product_recommendations[disease]["benefit"]

    return disease, confidence, product, benefit