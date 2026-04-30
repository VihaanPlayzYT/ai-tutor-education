import streamlit as st
import joblib
import google.generativeai as genai

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**Subject Prediction + Powerful Answers from Gemini**")

# Load ML model for subject prediction
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
    return genai.GenerativeModel('gemini-1.5-flash')   # fast and good for tutoring

gemini_model = configure_gemini()

# User Input
question = st.text_input("Enter your academic question:", 
                        placeholder="e.g., Explain ionic and covalent bonding")

if st.button("🧠 Get Gemini Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Gemini is thinking..."):
        # 1. Predict subject using ML model
        vec = vectorizer.transform([question])
        predicted_subject = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()

        # 2. Call Gemini
        prompt = f"""You are an excellent, patient academic tutor.
Student's question: "{question}"

Predicted subject: {predicted_subject} (confidence: {confidence:.1%})

Provide a clear, accurate, and educational answer. 
Explain step by step. Use simple language suitable for high school / college students.
Include examples if helpful."""

        try:
            response = gemini_model.generate_content(prompt)
            gemini_answer = response.text

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Predicted Subject:** {predicted_subject}")
            with col2:
                st.info(f"**Confidence:** {confidence:.1%}")

            st.markdown("### 📝 Gemini's Answer")
            st.markdown(gemini_answer)

        except Exception as e:
            st.error(f"Error with Gemini API: {e}")
            st.info("Make sure your Gemini API key is correctly added in Streamlit Secrets.")

# Optional: Show only subject prediction
if st.button("🔍 Predict Subject Only"):
    if question.strip():
        vec = vectorizer.transform([question])
        subject = model.predict(vec)[0]
        conf = model.predict_proba(vec).max()
        st.success(f"**Subject:** {subject} | Confidence: {conf:.1%}")
    else:
        st.warning("Enter a question first.")

st.caption("AI Tutor • ML Subject Prediction + Google Gemini 1.5 Flash")
