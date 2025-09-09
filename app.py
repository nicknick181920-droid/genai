


import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load IBM Granite Model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-2b-base")
    model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.1-2b-base")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_model()

# Streamlit UI
st.set_page_config(page_title="Personal Finance Chatbot", page_icon="💰", layout="wide")
st.title("💬 Personal Finance Chatbot")
st.markdown("### Get intelligent guidance for **savings, taxes, and investments**")

# User Type (for tone)
user_type = st.selectbox("Who are you?", ["Student", "Professional"])

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User Input
user_input = st.text_input("Ask me about your finances:")

if user_input:
    # Adjust prompt tone
    if user_type == "Student":
        prompt = f"Explain in simple terms for a student: {user_input}"
    else:
        prompt = f"Give a detailed, professional explanation: {user_input}"

    response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]

    # Save chat
    st.session_state.history.append({"user": user_input, "bot": response})

# Show chat
for chat in st.session_state.history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")
    st.write("---")
