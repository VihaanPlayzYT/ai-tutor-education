import streamlit as st
import joblib
import google.generativeai as genai

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**Smart Academic Helper** — Subject Prediction + Answers from Gemini")

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
def configure_gemini():
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    return genai.GenerativeModel('gemini-2.5-flash')

gemini_model = configure_gemini()

# Input
question = st.text_input(
    "Enter your academic question:", 
    placeholder="e.g., Solve 2x + 5 = 15 or Explain ionic and covalent bonding"
)

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict Subject", type="secondary")
with col2:
    gemini_btn = st.button("🧠 Get Gemini Answer", type="primary")

# Predict Subject Only
if predict_btn:
    if question.strip():
        with st.spinner("Predicting..."):
            vec = vectorizer.transform([question])
            subject = model.predict(vec)[0]
            confidence = model.predict_proba(vec).max()
            st.success(f"**Predicted Subject:** {subject}")
            st.info(f"**Confidence:** {confidence:.1%}")
    else:
        st.warning("Please enter a question.")

# Get Gemini Answer - Full Width
if gemini_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gemini is thinking..."):
            # Subject Prediction
            vec = vectorizer.transform([question])
            predicted_subject = model.predict(vec)[0]
            confidence = model.predict_proba(vec).max()

            # Call Gemini
            prompt = f"""You are an excellent academic tutor.
Student's question: "{question}"

Predicted subject: {predicted_subject}

Give a clear, detailed, and educational answer. Explain step by step. Use simple language suitable for students."""

            try:
                response = gemini_model.generate_content(prompt)
                gemini_answer = response.text

                # Display Subject Info
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"**Subject:** {predicted_subject}")
                with col_b:
                    st.info(f"**Confidence:** {confidence:.1%}")

                # Full Width Answer
                st.markdown("### 📝 Gemini's Answer")
                st.markdown(gemini_answer)

            except Exception as e:
                st.error(f"Error with Gemini: {str(e)}")

# Footer
st.caption("AI Tutor • ML Subject Prediction + Google Gemini 2.5 Flash")
