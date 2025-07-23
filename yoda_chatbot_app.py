
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import random
import os
import base64

# Page config
st.set_page_config(
    page_title="Yoda Bot - Jedi Wisdom Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown('''<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #00ff00;
        border: 2px solid #00ff00;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .yoda-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .yoda-avatar {
        text-align: center;
        font-size: 4rem;
        margin: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .emotion-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .emotion-positive { background-color: #4CAF50; color: white; }
    .emotion-negative { background-color: #f44336; color: white; }
    .emotion-neutral { background-color: #2196F3; color: white; }
</style>''', unsafe_allow_html=True)

# Load model and components (cached)
@st.cache_resource
def load_yoda_components():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./yoda-chatbot-cornell"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return model, tokenizer, sentiment_analyzer, device

# Emotion detection function
def detect_emotion(text, sentiment_analyzer):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.1:
        return 'positive', compound
    elif compound <= -0.1:
        return 'negative', compound
    else:
        return 'neutral', compound

# Yoda rephraser
import re

def rephrase_yoda(sentence):
    sentence = sentence.strip()
    if not sentence:
        return ""

    # Basic clause split
    parts = re.split(r'[\.\?!]', sentence)[0].split()
    if len(parts) < 4:
        return sentence + ", yes."

    # Find subject (assume 1st word), verb (assume 2nd or 3rd)
    # Move object or remainder to front
    subject = parts[0]
    verb = parts[1] if len(parts) > 2 else ""
    rest = parts[2:]

    # Construct Yoda-style
    yoda_sentence = f"{' '.join(rest)}, {subject} {verb}".strip().capitalize() + ", hmmm."
    return yoda_sentence


# Response generation function
def generate_yoda_response(user_input, model, tokenizer, device):
    prompt = f"<|startoftext|>Human: {user_input}<|sep|>Yoda:"
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 80,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("<|endoftext|>")[0]
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    yoda_response = full_response.split("Yoda:")[-1].strip()
    if "<|endoftext|>" in yoda_response:
        yoda_response = yoda_response.split("<|endoftext|>")[0].strip()
    return yoda_response

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'emotions' not in st.session_state:
    st.session_state.emotions = []

# Load components
try:
    model, tokenizer, sentiment_analyzer, device = load_yoda_components()
    model_loaded = True
except:
    model_loaded = False
    st.error("Model not found. Please run the training script first.")

# Title
st.title("Yoda Bot - Jedi Wisdom Chatbot")

# Show image if uploaded
image_path = "yoda.png"
if os.path.exists(image_path):
    st.image(image_path, width=200, caption="Master Yoda", use_column_width=False)
else:
    st.warning("Yoda image not found. Please upload yoda.png.")

st.markdown('<div class="yoda-avatar">🧙</div>', unsafe_allow_html=True)
st.markdown('''
<div style="text-align: center; color: #00ff00; font-size: 1.2rem; margin-bottom: 2rem;">
    "Judge me by my size, do you? And well you should not. For my ally is the Force, and a powerful ally it is."
</div>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About Master Yoda")
    st.write("This chatbot channels the wisdom of Master Yoda to help you with: Self-reflection and introspection, Emotional guidance, Life decisions")
    st.header("Emotional State")
    if st.session_state.emotions:
        latest_emotion = st.session_state.emotions[-1]
        emotion_class = f"emotion-{latest_emotion[0]}"
        st.markdown(f'<div class="{emotion_class}">{latest_emotion[0].title()} ({latest_emotion[1]:.2f})</div>', unsafe_allow_html=True)

# Display chat history
for i, (msg, emotion) in enumerate(zip(st.session_state.messages, st.session_state.emotions)):
    role = "user-message" if i % 2 == 0 else "yoda-message"
    speaker = "You" if i % 2 == 0 else "Master Yoda"
    st.markdown(f'''
    <div class="chat-message {role}">
        <strong>{speaker}:</strong> {msg}
    </div>
    ''', unsafe_allow_html=True)

# Handle message
def handle_submit():
    user_input = st.session_state.get("user_input", "")
    if user_input and model_loaded:
        st.session_state.messages.append(user_input)
        emotion, score = detect_emotion(user_input, sentiment_analyzer)
        st.session_state.emotions.append((emotion, score))
        with st.spinner("Master Yoda is consulting the Force..."):
            time.sleep(1)
            raw_reply = generate_yoda_response(user_input, model, tokenizer, device)
            yoda_reply = rephrase_yoda(raw_reply)
        st.session_state.messages.append(yoda_reply)
        st.session_state.emotions.append(('neutral', 0))
        st.session_state.user_input = ""  # Clear input field (safe here in callback)

# Input
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.text_input(
        "Speak with Master Yoda...",
        key="user_input",
        placeholder="Young padawan, what troubles you?",
        on_change=handle_submit
    )

# Footer
st.markdown('<div style="text-align: center; margin-top: 3rem; color: #666; font-style: italic;">"Do or do not, there is no try." - Master Yoda</div>', unsafe_allow_html=True)
