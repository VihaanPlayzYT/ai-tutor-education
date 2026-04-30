import streamlit as st
import joblib

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("Ask any question and I'll predict the **subject**!")

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ai_tutor_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except:
        st.error("Model files not found. Please make sure ai_tutor_model.pkl and vectorizer.pkl are in the app folder.")
        st.stop()

model, vectorizer = load_model()

# Input
question = st.text_input("Enter your question:", placeholder="e.g., Solve 2x + 5 = 15")

if st.button("Predict Subject", type="primary"):
    if question.strip():
        with st.spinner("Analyzing question..."):
            question_vec = vectorizer.transform([question])
            subject = model.predict(question_vec)[0]
            confidence = model.predict_proba(question_vec).max()
            
            # Display result
            st.success(f"**Predicted Subject:** {subject}")
            st.info(f"**Confidence:** {confidence:.1%}")
            
            if confidence < 0.4:
                st.warning("⚠️ Low confidence. The model is still learning.")
    else:
        st.warning("Please enter a question.")

# Footer
st.caption("AI Tutor • Trained on Physics, Math, Chemistry, Computer Science & English")
