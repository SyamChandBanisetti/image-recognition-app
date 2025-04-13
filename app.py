import streamlit as st
from PIL import Image
from utils import predict_image

st.set_page_config(page_title="Image Recognition App", layout="centered")
st.title("🔍 Image Recognition & Understanding App")

st.markdown("Upload an image and the model will predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Analyzing Image..."):
        preds = predict_image(img)

    st.success("✅ Prediction Complete!")
    for i, pred in enumerate(preds):
        st.write(f"**{i+1}. {pred[1]}** — {round(pred[2]*100, 2)}% confidence")
