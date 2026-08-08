import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")

st.write("Upload an MRI scan and choose a model.")

# =========================
# MODEL OPTIONS
# =========================
model_options = {
    "Binary - VGG16": "binary_vgg16.h5",
    "Binary - Custom CNN": "binary_custom_cnn.h5",
    "Binary - ResNet50": "binary_resnet50.h5",
    "MultiClass - Custom CNN": "multiclass_custom_cnn.h5",
    "MultiClass - ResNet50": "multiclass_resnet50.h5",
    "MultiClass - VGG16": "multiclass_vgg16.h5"
}

selected_model = st.selectbox(
    "Choose Model",
    list(model_options.keys())
)

model_path = model_options[selected_model]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_path)

# =========================
# CLASS NAMES
# =========================
binary_classes = ["No Tumor", "Tumor"]

multiclass_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image):

    img = np.array(image)

    img = cv2.resize(img, (224, 224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    processed_image = preprocess_image(image)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(processed_image)

    # =========================
    # BINARY MODELS
    # =========================
    if "Binary" in selected_model:

        confidence = float(prediction[0][0])

        if confidence > 0.5:
            predicted_class = binary_classes[1]
            final_confidence = confidence
        else:
            predicted_class = binary_classes[0]
            final_confidence = 1 - confidence

    # =========================
    # MULTICLASS MODELS
    # =========================
    else:

        predicted_index = np.argmax(prediction)

        predicted_class = multiclass_names[predicted_index]

        final_confidence = np.max(prediction)

    # =========================
    # RESULTS
    # =========================
    st.subheader("Prediction")

    st.success(f"Result: {predicted_class}")

    st.metric(
        "Confidence",
        f"{final_confidence * 100:.2f}%"
    )