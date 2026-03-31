import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 🔥 Page config (TOP पर होना जरूरी)
st.set_page_config(page_title="Brain Cell AI", layout="centered")

# 🔥 Title UI
st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>
    🧠 Brain Cell Classification System
    </h1>
""", unsafe_allow_html=True)

st.write("Upload a microscopy image to classify brain cell type")

# Load model
model = load_model("models/brain_cell_classifier.h5")

classes = ["Astrocyte","Neuron","NSC","Oligodendrocyte"]

# Upload
uploaded_file = st.file_uploader("📂 Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    # Save image
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp.png", caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.load_img("temp.png", target_size=(224,224))
    img_array = image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 Button (prediction trigger)
    if st.button("🔍 Predict"):

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)

        result = classes[class_index]
        confidence = np.max(prediction)

        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence:.4f}")

        # 🔥 Heatmap
        heatmap = np.mean(img_array[0], axis=-1)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.imread("temp.png")
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

        output = heatmap * 0.4 + original

        cv2.imwrite("output.png", output)

        st.image("output.png", caption="🔥 AI Focus Heatmap", use_container_width=True)