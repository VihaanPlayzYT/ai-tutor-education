import streamlit as st
import joblib
import google.generativeai as genai
from datetime import datetime

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**Smart Academic Helper** — ML Subject Prediction + Gemini AI with Chat History")

# Load ML Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ai_tutor_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except:
        st.error("Model files not found!")
        st.stop()

model, vectorizer = load_model()

# Configure Gemini
@st.cache_resource
def configure_gemini(model_name):
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    return genai.GenerativeModel(model_name)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.5-flash"

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Choose Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0 if st.session_state.selected_model == "gemini-2.5-flash" else 1
    )
    st.session_state.selected_model = model_choice
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Input
question = st.text_input(
    "Enter your academic question:", 
    placeholder="e.g., Solve 2x + 5 = 15 or Explain Newton's laws"
)

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict Subject", type="secondary")
with col2:
    gemini_btn = st.button("🧠 Get Gemini Answer", type="primary")

# Predict Subject Only
if predict_btn and question.strip():
    with st.spinner("Predicting subject..."):
        vec = vectorizer.transform([question])
        subject = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        st.success(f"**Predicted Subject:** {subject}")
        st.info(f"**Confidence:** {confidence:.1%}")

# Get Gemini Answer with Chat History
if gemini_btn and question.strip():
    with st.spinner("Gemini is thinking..."):
        # Subject Prediction
        vec = vectorizer.transform([question])
        predicted_subject = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()

        # Build prompt with context
        prompt = f"""You are an excellent academic tutor.

Student's question: "{question}"

The ML model predicted this belongs to **{predicted_subject}** subject with {confidence:.1%} confidence.

Answer clearly and step by step. Use examples when helpful.
Use LaTeX for math equations (e.g., $x = 5$)."""

        try:
            gemini_model = configure_gemini(st.session_state.selected_model)
            response = gemini_model.generate_content(prompt)
            gemini_answer = response.text

            # Add to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": question,
                "subject": predicted_subject,
                "confidence": confidence
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": gemini_answer
            })

        except Exception as e:
            st.error(f"Gemini Error: {e}")

# Display Chat History
if st.session_state.messages:
    st.markdown("### 📜 Conversation History")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
            st.caption(f"Predicted: {msg.get('subject', '')} | Confidence: {msg.get('confidence', 0):.1%}")
        else:
            st.markdown(f"**Gemini:**")
            st.markdown(msg["content"])
            st.divider()

# Footer
st.caption("AI Tutor • Dataset ML Model + Google Gemini | Chat History & LaTeX Support")
