import streamlit as st
import joblib
import re

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

st.title("🎓 AI Tutor")
st.markdown("Ask me **any academic question** — I'll try to identify the subject and give you a helpful answer!")

# Load model and vectorizer
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

# Simple rule-based answers for common questions
def get_answer(question, subject):
    q = question.lower()
    
    if subject == "Math":
        if "solve for x" in q or "2x + 5 = 15" in q:
            return "To solve **2x + 5 = 15**: Subtract 5 from both sides → 2x = 10, then divide by 2 → **x = 5**."
        if "area of a circle" in q or "radius" in q:
            return "The area of a circle is **πr²**. If radius = 5, area = π × 25 ≈ **78.54 square units**."
        return "This looks like a Math question. Could you show me the full problem so I can solve it step by step?"
    
    elif subject == "Physics":
        if "newton" in q:
            return "Newton's Three Laws:\n1. **Inertia**: An object stays at rest or in motion unless acted upon by a force.\n2. **F=ma**: Force = mass × acceleration.\n3. **Action-Reaction**: For every action, there is an equal and opposite reaction."
        return "This is a Physics question. Tell me more details so I can explain the concept clearly."
    
    elif subject == "Chemistry":
        if "ionic" in q or "covalent" in q:
            return "**Ionic bonding**: Electrons are transferred (usually between metal and non-metal).\n**Covalent bonding**: Electrons are shared (usually between non-metals)."
        if "photosynthesis" in q:
            return "Photosynthesis: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ (in presence of sunlight and chlorophyll)."
        return "Chemistry question detected. What specific part would you like explained?"
    
    elif subject == "Computer Science":
        if "declare a variable" in q or "python" in q:
            return "In Python, you declare a variable like this:\n```python\nname = 'Vihaan'\nage = 16\n```"
        if "hello world" in q:
            return "Here's the classic program:\n```python\nprint('Hello, World!')\n```"
        return "This seems like a programming question. What language or concept are you working with?"
    
    else:  # English or unknown
        if "there" in q and "their" in q:
            return "**There** = place (over there)\n**Their** = possession (their book)\n**They're** = they are"
        return "This appears to be an English or language question. How can I help you with grammar, vocabulary, or writing?"

# Main Interface
question = st.text_input("Enter your question:", placeholder="e.g., Explain ionic and covalent bonding", key="question")

if st.button("Get Answer", type="primary"):
    if question.strip():
        with st.spinner("Thinking..."):
            # Predict subject
            question_vec = vectorizer.transform([question])
            subject = model.predict(question_vec)[0]
            confidence = model.predict_proba(question_vec).max()

            # Get helpful answer
            answer = get_answer(question, subject)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Subject:** {subject}")
            with col2:
                st.info(f"**Confidence:** {confidence:.1%}")

            st.markdown("### 💡 Answer")
            st.write(answer)
            
            if confidence < 0.6:
                st.warning("⚠️ Note: The subject prediction has low confidence. The answer is based on keyword matching.")
    else:
        st.warning("Please enter a question.")

st.caption("AI Tutor • Trained on 491 questions | Physics, Math, Chemistry, CS & English")
