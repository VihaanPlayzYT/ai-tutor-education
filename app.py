import streamlit as st
import joblib
import google.generativeai as genai

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**Smart Academic Helper** — Subject Prediction + Answers from Gemini")

# Load ML Model for subject prediction
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ai_tutor_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except:
        st.error("Model files not found! Please make sure ai_tutor_model.pkl and vectorizer.pkl exist.")
        st.stop()

model, vectorizer = load_model()

# Configure Gemini
@st.cache_resource
def configure_gemini():
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    # Updated model name (2026)
    return genai.GenerativeModel('gemini-2.5-flash')

gemini_model = configure_gemini()

# User Input
question = st.text_input(
    "Enter your academic question:", 
    placeholder="e.g., Explain ionic and covalent bonding or Solve 2x + 5 = 15"
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔍 Predict Subject", type="secondary"):
        if question.strip():
            with st.spinner("Predicting subject..."):
                vec = vectorizer.transform([question])
                subject = model.predict(vec)[0]
                confidence = model.predict_proba(vec).max()
                st.success(f"**Predicted Subject:** {subject}")
                st.info(f"**Confidence:** {confidence:.1%}")
        else:
            st.warning("Please enter a question.")

with col2:
    if st.button("🧠 Get Gemini Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Gemini is thinking..."):
            # Predict subject
            vec = vectorizer.transform([question])
            predicted_subject = model.predict(vec)[0]
            confidence = model.predict_proba(vec).max()

            # Prompt for Gemini
            prompt = f"""You are an excellent, patient, and clear academic tutor.
Student asked: "{question}"

Predicted subject: {predicted_subject} (confidence: {confidence:.1%})

Give a clear, accurate, step-by-step explanation suitable for high school or early college students.
Use examples when helpful. Keep the language simple and educational."""

            try:
                response = gemini_model.generate_content(prompt)
                gemini_answer = response.text

                # Display Results
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"**Subject:** {predicted_subject}")
                with col_b:
                    st.info(f"**Confidence:** {confidence:.1%}")

                st.markdown("### 📝 Gemini's Answer")
                st.markdown(gemini_answer)

            except Exception as e:
                st.error(f"Error with Gemini: {e}")
                st.info("Please check if your Gemini API key is correctly added in Streamlit Secrets.")

# Footer
st.caption("AI Tutor • ML Subject Prediction + Google Gemini 2.5 Flash")
