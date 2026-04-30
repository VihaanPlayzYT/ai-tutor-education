import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎓 AI Tutor")
st.write("Ask me anything! I'll try to predict the subject.")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('ai_tutor_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

question = st.text_input("Type your question here:")

if st.button("Get Subject"):
    if question:
        question_vec = vectorizer.transform([question])
        prediction = model.predict(question_vec)[0]
        probability = model.predict_proba(question_vec).max()
        
        st.success(f"**Predicted Subject:** {prediction}")
        st.info(f"Confidence: {probability:.1%}")
    else:
        st.warning("Please enter a question.")
