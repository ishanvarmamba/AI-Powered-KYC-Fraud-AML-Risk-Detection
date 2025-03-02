import os
import json
import base64
import streamlit as st
from google.oauth2 import service_account
from google.cloud import documentai_v1 as documentai
from PIL import Image
import pdf2image
import numpy as np
import pytesseract
import openai
import pandas as pd
import re

# ------------------------ 1️⃣ Load API Keys ------------------------
st.set_page_config(page_title="KYC AI Fraud & AML Risk Detection", layout="wide")

# ✅ Load Google Cloud API credentials from Streamlit Secrets
gcp_credentials = json.loads(st.secrets["gcp"]["credentials"])
credentials = service_account.Credentials.from_service_account_info(gcp_credentials)

# ✅ Initialize OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Initialize Google Cloud Document AI Client
document_client = documentai.DocumentUnderstandingServiceClient(credentials=credentials)

# ------------------------ 2️⃣ Helper Functions ------------------------
def extract_text_from_pdf(pdf_path):
    """Converts PDF pages to images and extracts text using OCR."""
    images = pdf2image.convert_from_path(pdf_path)
    extracted_text = ""
    for img in images:
        text = pytesseract.image_to_string(img)
        extracted_text += text + "\n"
    return extracted_text.strip()

def encode_image(image_path):
    """Encodes an image as base64 for Google Cloud Document AI."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_kyc_document(file_path, file_type):
    """Extracts KYC details using Google Cloud Document AI."""
    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(file_path)
    else:
        base64_img = encode_image(file_path)
        document = {
            "content": base64_img,
            "mime_type": "image/png" if file_type == "png" else "image/jpeg",
        }
        request = documentai.ProcessRequest(
            name=f"projects/{gcp_credentials['project_id']}/locations/us/processors/YOUR_PROCESSOR_ID",
            raw_document=document,
        )
        response = document_client.process_document(request=request)
        extracted_text = response.document.text

    return extracted_text.strip()

def analyze_fraud_risk(kyc_text):
    """Analyzes fraud risk using OpenAI GPT-4o."""
    prompt = f"""
    Analyze the following KYC document for fraud risk:

    {kyc_text}

    - Identify inconsistencies in document details.
    - Signs of tampering, duplication, or forgery.
    - Unusual patterns indicating fraud.
    - Provide a fraud risk percentage (0-100%).
    - Provide a short explanation.

    Return output in this format:
    **Fraud Risk:** X%  
    **Analysis:** Explanation here.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a fraud risk analyst."}, {"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response["choices"][0]["message"]["content"].strip()

def analyze_aml_risk(kyc_text):
    """Analyzes AML risk using OpenAI GPT-4o."""
    prompt = f"""
    Analyze the following KYC document for Anti-Money Laundering (AML) risk:

    {kyc_text}

    - Check for high-risk nationalities (sanctions, high-corruption index).
    - Identify Politically Exposed Persons (PEPs).
    - Detect duplicate or fake identities.
    - Transactions linked to financial crime.
    - Provide an AML risk percentage (0-100%).
    - Provide a short explanation.

    Return output in this format:
    **AML Risk:** X%  
    **Analysis:** Explanation here.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AML risk analyst."}, {"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response["choices"][0]["message"]["content"].strip()

# ------------------------ 3️⃣ Streamlit UI ------------------------
st.title("🔍 KYC AI: Fraud & AML Risk Detection")
st.subheader("📑 Upload a KYC document (Passport, ID, License, etc.)")

uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "png", "pdf"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    with open(f"temp.{file_extension}", "wb") as f:
        f.write(uploaded_file.getvalue())
        file_path = f.name

    st.info("🔍 Extracting KYC details...")
    kyc_text = analyze_kyc_document(file_path, file_extension)
    st.success("✅ KYC extraction successful!")

    # Display Extracted KYC Details
    kyc_data = [line.split(": ") for line in kyc_text.split("\n") if ": " in line]
    kyc_df = pd.DataFrame(kyc_data, columns=["Field", "Value"])
    st.subheader("📜 Extracted KYC Details")
    st.table(kyc_df)

    # Fraud Risk Analysis
    st.info("🔍 Running fraud risk analysis...")
    fraud_analysis = analyze_fraud_risk(kyc_text)

    # AML Risk Analysis
    st.info("🔍 Running AML risk analysis...")
    aml_analysis = analyze_aml_risk(kyc_text)

    # Extract Percentages
    def extract_percentage(text):
        match = re.search(r'(\d{1,3})%', text)
        return match.group(1) if match else "Unknown"

    fraud_risk = extract_percentage(fraud_analysis)
    aml_risk = extract_percentage(aml_analysis)

    # Display Fraud & AML Risk
    st.subheader("⚠️ Risk Assessment")
    st.write(f"**Fraud Risk Level:** {fraud_risk}%")
    st.write(f"**AML Risk Level:** {aml_risk}%")

    st.subheader("📌 Fraud Analysis")
    st.text(fraud_analysis)

    st.subheader("📌 AML Analysis")
    st.text(aml_analysis)

    os.remove(file_path)