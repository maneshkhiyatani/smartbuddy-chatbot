# aimport os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login

# Login with Hugging Face token (from Streamlit Secrets)
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Load model and tokenizer
model_name = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)

# App UI
st.set_page_config(page_title="SmartBuddy Chatbot", page_icon="🤖")
st.title("🤖 SmartBuddy - Multilingual Chatbot")

st.markdown("Type your message below and get a smart reply in English, Urdu, Hindi, or Roman Urdu.")

# Chat input
user_input = st.text_input("You:", placeholder="Ask me anything...")

# Chat response logic
def chatbot_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Show response
if user_input:
    with st.spinner("Thinking..."):
        reply = chatbot_response(user_input)
    st.success("SmartBuddy:")
    st.write(reply)

