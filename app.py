# app.py

import streamlit as st
from smartbuddy_model import chatbot_response

st.set_page_config(page_title="SmartBuddy Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 SmartBuddy Chatbot")
st.markdown("Ask me anything in English, Roman Urdu, or Hindi!")

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip() != "":
    with st.spinner("SmartBuddy is typing..."):
        response = chatbot_response(user_input)
        st.text_area("SmartBuddy:", value=response, height=200)
