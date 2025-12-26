import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Face Mask Detection (YOLOv8)",
    layout="centered"
)

st.title("Face Mask Detection — YOLOv8")
st.write("Upload an image to detect face mask compliance.")

@st.cache_resource
def load_model():
    return YOLO("artifacts/yolov8_face_mask/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        results = model(np.array(image))

    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_container_width=True)
