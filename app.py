import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ================= LOAD MODEL =================
model = load_model(r"C:\Users\yasht\Desktop\CV_FINAL\defect_classifier.h5")

classes = ["inclusion", "rolled_pit", "silk_spot", "waist folding", "crack"]

# ================= FUNCTION =================
def process_image(image):

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 80, 180)

    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(edges)
    crack_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 150:
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = max(w,h) / (min(w,h)+1)

            if aspect_ratio > 5 and area < 2000:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                crack_count += 1

    # ================= ML =================
    resized = cv2.resize(edges, (128,128))
    input_img = resized.reshape(1,128,128,1) / 255.0

    pred = model.predict(input_img, verbose=0)
    class_id = np.argmax(pred)

    label = classes[class_id]
    confidence = np.max(pred)

    # ================= DEFECT % =================
    defect_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percentage = (defect_pixels / total_pixels) * 100

    # ================= FINAL DECISION =================
    if crack_count >= 3 and percentage > 5:
        label = "crack"

    # ================= OUTPUT =================
    output = img.copy()

    if label == "crack":
        red_overlay = np.zeros_like(img)
        red_overlay[:,:,2] = 255

        output = np.where(
            cv2.merge([mask,mask,mask]) > 0,
            cv2.addWeighted(output, 0.5, red_overlay, 0.5, 0),
            output
        )

    return output, label, confidence, percentage


# ================= UI =================
st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("🔍 Metal Defect Detection System")
st.write("Upload an image to detect defects (Crack, Inclusion, etc.)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    if st.button("Detect Defect"):

        output, label, confidence, percentage = process_image(image)

        st.subheader("Result")

        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.markdown(f"### 🧠 Defect: `{label}`")
        st.markdown(f"### 📊 Confidence: `{confidence:.2f}`")
        st.markdown(f"### 📉 Defect %: `{percentage:.2f}`")