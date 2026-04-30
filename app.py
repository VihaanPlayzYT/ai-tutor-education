import streamlit as st
import joblib

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("**Smart Academic Helper** — Predicts subject + gives real AI answers")

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

# Input
question = st.text_input("Enter your question:", 
                        placeholder="e.g., Explain ionic and covalent bonding", 
                        key="q_input")

col1, col2 = st.columns(2)

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
    if st.button("🧠 Get AI Answer", type="primary"):
        if question.strip():
            with st.spinner("Getting detailed answer from AI..."):
                # Here we simulate calling a powerful AI (Grok/Claude)
                # In real deployment, you would call Groq, OpenAI, or Anthropic API
                
                st.info("**AI Answer:**")
                
                # For now, we'll use a placeholder. Later we can integrate real API
                st.write("**Note:** To enable real Grok/Claude answers, we need to add API integration.")
                st.caption("Would you like me to add **Groq** (fast & free) or **OpenAI** integration?")

                # Simple fallback answer
                st.write("---")
                st.write("**Subject-based explanation:**")
                st.write("I'm currently using rule-based + ML subject detection. For high-quality answers, we can integrate Grok or Claude API.")
                
        else:
            st.warning("Please enter a question first.")

# Current limitations note
st.caption("AI Tutor v2 • Subject prediction using ML | Real AI answer coming soon")
