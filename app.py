import streamlit as st
import joblib

st.title("🎓 AI Tutor")
st.write("Ask any question and I'll predict the subject!")

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("ai_tutor_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

question = st.text_input("Enter your question:")

if st.button("Predict Subject"):
    if question.strip():
        vec = vectorizer.transform([question])
        subject = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        
        st.success(f"**Predicted Subject:** {subject}")
        st.info(f"Confidence: {confidence:.1%}")
    else:
        st.warning("Please type a question.")
