import streamlit as st
import openai
import os
import base64
from PIL import Image
import tempfile
import pandas as pd
import re  # ✅ For extracting fraud/AML percentages
from google.cloud import documentai
from google.oauth2 import service_account
import json  # Import the json module

# ------------------------ 1️⃣ Load API Keys ------------------------
# Streamlit Page Configuration
st.set_page_config(page_title="KYC AI Fraud & AML Risk Detection", layout="wide")

# Initialize Google Cloud Document AI Client
try:
    # Attempt to load credentials from Streamlit secrets
    gcp_credentials = st.secrets["gcp"]["credentials"]

    # Check if credentials are a string (if so, parse as JSON)
    if isinstance(gcp_credentials, str):
        gcp_credentials = json.loads(gcp_credentials)

    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    document_client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    processor_id = st.secrets["gcp"]["processor_id"]  # Load processor ID from secrets
except KeyError as e:
    st.error(f"Missing secret key: {e}. Check your secrets.toml file.")
    st.stop()

# ✅ Initialize OpenAI API
openai.api_key = st.secrets["openai"]["api_key"]

# ✅ Function to extract text using Google Document AI
def extract_text_google(file_path, mime_type, processor_name):
    """Extracts text from a document using Google Cloud Document AI."""
    try:
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()

        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=processor_name,  # Use the processor_name parameter
            raw_document=raw_document
        )

        response = document_client.process_document(request=request)
        return response.document.text
    except Exception as e:
        st.error(f"Error extracting text with Google Document AI: {e}")
        return ""

# ✅ Function to extract key KYC details automatically
def extract_kyc_details(extracted_text):
    try:
        response = openai.chat.completions.create(  # Use openai.chat.completions
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an AI specialized in KYC document verification. Extract structured details such as Name, Date of Birth, ID Type, ID Number, Address, Nationality, Issued Date, and Expiry Date."},
                {"role": "user",
                 "content": f"Extract key KYC details from the following document text:\n\n{extracted_text}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()  # Access response correctly
    except Exception as e:
        st.error(f"Error during OpenAI KYC extraction: {e}")
        return "Error occurred during KYC extraction."

# ✅ Function to extract percentage from GPT response
def extract_percentage(text):
    match = re.search(r'(\d{1,3})%', text)
    return match.group(1) if match else "Unknown"

# ✅ Function for fraud risk analysis
def analyze_fraud_risk(kyc_text):
    fraud_prompt = f"""
    Analyze the following KYC details for potential fraud risk:

    {kyc_text}

    Identify:
    - Inconsistencies in document details
    - Signs of tampering or forgery
    - Unusual patterns that indicate fraud
    - Provide a fraud risk percentage (0-100%)
    - Provide a short explanation of the risk assessment.
    """

    try:
        response = openai.chat.completions.create(  # Use openai.chat.completions
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an AI fraud detection expert analyzing KYC documents for financial services."},
                {"role": "user", "content": fraud_prompt}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content.strip()  # Access response correctly
    except Exception as e:
        st.error(f"Error during OpenAI Fraud Analysis: {e}")
        return "Error occurred during fraud analysis."

# ✅ Function for AML risk analysis
def analyze_aml_risk(kyc_text):
    aml_prompt = f"""
    Analyze the following KYC details for AML (Anti-Money Laundering) risks:

    {kyc_text}

    Identify:
    - High-risk nationalities (sanctions, high-corruption index)
    - Politically Exposed Persons (PEPs)
    - Duplicate or fake identities
    - Transactions linked to financial crime
    - Provide an AML risk percentage (0-100%)
    - Provide a short explanation of the AML risk.
    """

    try:
        response = openai.chat.completions.create(  # Use openai.chat.completions
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI specialized in AML risk assessment for financial services."},
                {"role": "user", "content": aml_prompt}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content.strip()  # Access response correctly
    except Exception as e:
        st.error(f"Error during OpenAI AML Analysis: {e}")
        return "Error occurred during AML analysis."

# ✅ Streamlit UI for file upload
st.title("🔍 AI-Based KYC Extraction (Google Document AI) + Fraud & AML (GPT-4o)")

uploaded_file = st.file_uploader("📂 Upload a KYC document (Passport, ID, License, etc.)", type=["jpg", "png", "pdf"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    mime_type = "application/pdf" if file_extension == "pdf" else "image/jpeg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    st.info("⏳ Extracting text using Google Document AI...")

    # Construct processor name here
    processor_name = f"projects/{st.secrets['gcp']['project_id']}/locations/us/processors/{st.secrets['gcp']['processor_id']}"

    # ✅ Extract text using Google Document AI
    extracted_text = extract_text_google(temp_path, mime_type, processor_name)

    # ✅ Extract KYC details
    st.info("⏳ Extracting structured KYC details using GPT-4o...")
    kyc_details = extract_kyc_details(extracted_text)
    st.success("✅ KYC extraction completed!")

    # ✅ Display extracted KYC details
    st.subheader("📄 Extracted KYC Details")
    st.text(kyc_details)

    # ✅ Run fraud & AML risk analysis
    st.info("⏳ Running fraud risk analysis...")
    fraud_analysis = analyze_fraud_risk(kyc_details)
    fraud_risk = extract_percentage(fraud_analysis)

    st.info("⏳ Running AML risk analysis...")
    aml_analysis = analyze_aml_risk(kyc_details)
    aml_risk = extract_percentage(aml_analysis)

    # ✅ Display Fraud & AML risk with percentages at the top
    st.subheader("⚠️ Risk Assessment")
    st.write(f"**Fraud Risk Level:** {fraud_risk}%")
    st.write(f"**AML Risk Level:** {aml_risk}%")

    st.subheader("📌 Fraud Analysis")
    st.text(fraud_analysis)

    st.subheader("📌 AML Analysis")
    st.text(aml_analysis)

    # Cleanup
    os.remove(temp_path)

