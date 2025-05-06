# smartbuddy_model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Chatbot response function
def chatbot_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)
