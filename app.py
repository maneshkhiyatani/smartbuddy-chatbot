import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import os

# Get token from Hugging Face Spaces secrets
hf_token = os.environ.get("HF_TOKEN")
login(hf_token)

# Load model and tokenizer
model_name = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)

# Streamlit UI
st.title("🤖 Smart Chatbot using LaMini-T5")

user_input = st.text_input("Ask something...")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("Answer:", response)




