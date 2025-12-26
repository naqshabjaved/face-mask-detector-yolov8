import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

MODEL_PATH = "artifacts/yolov8_face_mask/weights/best.pt"

st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

st.title("Face Mask Detection")
st.write("YOLOv8-based face mask detection (with / without / incorrect mask)")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    if st.button("Run Detection"):
        with st.spinner("Running inference..."):
            results = model(temp_path, conf=conf_threshold)
            annotated = results[0].plot()

        st.image(
            annotated,
            caption="Prediction Result",
            use_container_width=True
        )

        os.remove(temp_path)
