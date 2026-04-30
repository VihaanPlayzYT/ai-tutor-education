import streamlit as st
import joblib
import google.generativeai as genai

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**ML Subject Prediction + Gemini AI Answers**")

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

if st.button("🧠 Get Gemini Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Analyzing with AI Tutor..."):
        # Step 1: Predict subject using your dataset-trained model
        vec = vectorizer.transform([question])
        predicted_subject = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()

        # Step 2: Send to Gemini with subject context
        prompt = f"""You are an expert academic tutor.

Student's question: "{question}"

The AI model predicted this question belongs to **{predicted_subject}** subject with {confidence:.1%} confidence.

Provide a clear, accurate, and well-explained answer suitable for high school or early college students.
Explain concepts step by step. Use examples where helpful."""

        try:
            response = gemini_model.generate_content(prompt)
            gemini_answer = response.text

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Predicted Subject:** {predicted_subject}")
            with col2:
                st.info(f"**ML Confidence:** {confidence:.1%}")

            st.markdown("### 📝 Gemini's Answer")
            st.markdown(gemini_answer)

        except Exception as e:
            st.error(f"Gemini Error: {e}")

# Optional: Just predict subject
if st.button("🔍 Predict Subject Only"):
    if question.strip():
        vec = vectorizer.transform([question])
        subject = model.predict(vec)[0]
        conf = model.predict_proba(vec).max()
        st.success(f"**Subject:** {subject}")
        st.info(f"**Confidence:** {conf:.1%}")
    else:
        st.warning("Please enter a question.")

st.caption("AI Tutor • Dataset-trained ML Model + Google Gemini 2.5 Flash")
