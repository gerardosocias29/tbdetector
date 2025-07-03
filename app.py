import os

# ✅ Fix for Streamlit write permissions (important: must be first)
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["HOME"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp"

# 🧠 Imports
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# 📦 Load model
model = load_model('my_model.h5')

# 🎯 App title
st.title("🩺 TB Chest X-Ray Classifier with Grad-CAM")

# 📤 Upload X-ray
uploaded_file = st.file_uploader("Upload a Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ⏳ Show upload progress
progress = st.empty()
if uploaded_file is not None:
    for percent in range(0, 101, 10):
        time.sleep(0.03)
        progress.progress(percent)
    progress.empty()

# 🔧 Preprocess function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    return image

# 🔥 Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🚀 UI: Only run prediction when user clicks "Predict"
if uploaded_file:
    st.markdown("---")
    if st.button("🔍 Predict"):
        status = st.status("Running prediction...", expanded=True)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        status.write("✅ Image loaded and preprocessed.")
        img_input = preprocess_image(image)
        img_expanded = np.expand_dims(img_input, axis=0)

        prob = model.predict(img_expanded)[0][0]
        label = "Tuberculosis Detected 😷" if prob > 0.5 else "Normal Lungs ✅"
        status.write(f"✅ Model prediction complete: `{label}`")

        heatmap = make_gradcam_heatmap(img_expanded, model)
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        overlay = heatmap_colored * 0.4 + image[..., ::-1] / 255.0

        status.write("✅ Grad-CAM visualization ready.")
        status.update(label="Prediction complete!", state="complete")

        st.subheader(f"Prediction: {label}")
        st.write(f"🧠 Confidence: `{prob:.4f}`")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original X-ray", use_column_width=True)
        with col2:
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            st.image(overlay_uint8, caption="Grad-CAM Heatmap", use_column_width=True)
