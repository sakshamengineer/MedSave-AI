import streamlit as st
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from StoreLocator import locatestore
from APICalling import Generatetxt

def prompt(title,ocr_text,result):
    return f"""
    You are an expert Indian pharmacist.

    {title}

    {ocr_text}:
    {result}

    Your task:

    1. Identify the medicine.
    2. Find its Jan Aushadhi equivalent.
    3. Find the approximate MRP of the brand medicine.
    4. Find the approximate Jan Aushadhi price.
    5. Calculate savings amount.
    6. Calculate savings percentage.

    Return In this format as it is like all enters , comas etc so that output looks clean.

        "brand_medicine": "",
        "equivalent": "",
        "brand_price": 0 (in rs),
        "jan_aushadhi_price": 0 (in rs),
        "savings": 0 (in rs),
        "savings_percentage": 0 (in %)

    Do NOT explain anything.
    Do NOT use markdown.
    """


@st.cache_resource
def load_model():
    return PaddleOCR(
    use_angle_cls=True,
    lang="en",
    enable_mkldnn=False,
    )

ocr = load_model()

st.set_page_config(
    page_title="MedSave AI",
)
st.title("MedSave AI")
st.subheader("Find Generetic Medicine")

but = st.radio(label="Choose a option",options = ["Upload Prescribsion","Upload Single Medicine","Type Manually"])

if but == "Upload Prescribsion" or but == "Upload Single Medicine":
    if but == "Upload Prescribsion":
        file = st.file_uploader("Upload Prescribsion",type=["jpg","jpeg","png","webp"])
    else:
        file = st.file_uploader("Upload Medicine",type=["jpg","jpeg","png","webp"])
    if file:
        image = Image.open(file)
        st.image(image,caption="Uploaded Image")
        max_size = 1200
        image.thumbnail((max_size, max_size))
        image_np = np.array(image)

        with st.spinner("Extracting text..."):

            result = ocr.predict(image_np)

            st.subheader("Detected Text")
            final_res = []
            for res in result:
                final_res.append(res["rec_texts"])

            st.write(Generatetxt(prompt("The following text has been extracted from a medicine strip using OCR.","Ocr_text",final_res)))

else:

    name = st.text_input("Enter Medicine Name ")
    Generate = st.button(label="Generate")
    if Generate and name:
        st.write(Generatetxt(prompt("I Have Given You The Name of Medicine","Medicine Name",name)))


st.title("Store Locator")
locatestore()