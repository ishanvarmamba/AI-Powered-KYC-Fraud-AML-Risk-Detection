import os
import json
import base64
import streamlit as st
from google.oauth2 import service_account
from google.cloud import documentai
from PIL import Image
import openai
import pandas as pd
import re

# ------------------------ 1️⃣ Load API Keys ------------------------
# Streamlit Page Configuration
st.set_page_config(page_title="KYC AI Fraud & AML Risk Detection", layout="wide")

# Initialize Google Cloud Document AI Client
try:
    # Load credentials from Streamlit secrets
    gcp_credentials = json.loads(st.secrets["gcp"]["credentials"])
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    document_client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    processor_id = st.secrets["gcp"]["processor_id"]  # Load processor ID from secrets
except KeyError as e:
    st.error(f"Missing secret key: {e}.  Check your secrets.toml file.")
    st.stop()

# ✅ Initialize OpenAI API
openai.api_key = st.secrets["openai"]["api_key"]

# ------------------------ 2️⃣ Helper Functions ------------------------
# def extract_text_from_pdf(pdf_path): # Remove this function
#     """Converts PDF pages to images and extracts text using OCR."""
#     images = pdf2image.convert_from_path(pdf_path)
#     extracted_text = ""
#     for img in images:
#         text = pytesseract.image_to_string(img)
#         extracted_text += text + "\n"
#     return extracted_text.strip()

def encode_image(image_path):
    """Encodes an image as base64 for Google Cloud Document AI."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def analyze_kyc_document(file_path, file_type):
    """Extracts KYC details using Google Cloud Document AI."""
    try:
        # Read the file content
        with open(file_path, "rb") as file:
            file_content = file.read()

        # Determine the correct MIME type
        mime_type = "application/pdf" if file_type == "pdf" else (
            "image/png" if file_type == "png" else "image/jpeg")

        # Create a raw document object
        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

        # Build the request
        request = documentai.ProcessRequest(
            name=f"projects/{gcp_credentials['project_id']}/locations/us/processors/{processor_id}",  # Replace with your processor ID
            raw_document=raw_document,
        )

        # Call the Document AI service
        response = document_client.process_document(request=request)

        # Extract the text from the response
        extracted_text = response.document.text

        return extracted_text.strip()

    except Exception as e:
        st.error(f"Error analyzing KYC document with Document AI: {e}")
        return ""  # Return an empty string in case of an error


def analyze_fraud_risk(kyc_text):
    """Analyzes fraud risk using OpenAI GPT-4o."""
    try:
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
    except Exception as e:
        st.error(f"Error during OpenAI Fraud Analysis: {e}")
        return "Fraud Risk: Unknown%\nAnalysis: An error occurred during analysis."

def analyze_aml_risk(kyc_text):
    """Analyzes AML risk using OpenAI GPT-4o."""
    try:
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
    except Exception as e:
        st.error(f"Error during OpenAI AML Analysis: {e}")
        return "AML Risk: Unknown%\nAnalysis: An error occurred during analysis."

def extract_percentage(text):
    """Extracts percentage from the analysis text."""
    match = re.search(r'(\d{1,3})%', text)
    return match.group(1) if match else "Unknown"

# ------------------------ 3️⃣ Streamlit UI ------------------------
st.title("🔍 KYC AI: Fraud & AML Risk Detection")
st.subheader("📑 Upload a KYC document (Passport, ID, License, etc.)")

uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "png", "pdf"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    file_path = f"temp.{file_extension}"  # Define file_path here

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        st.info("🔍 Extracting KYC details...")
        kyc_text = analyze_kyc_document(file_path, file_extension)
        st.success("✅ KYC extraction successful!")

        if not kyc_text:
            st.warning("Could not extract text from the document.")
            st.stop()

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

    finally:
        os.remove(file_path) # Ensure the temporary file is always deleted
