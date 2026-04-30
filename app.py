import streamlit as st
import joblib
import google.generativeai as genai

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: #f5f7ff;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(79,70,229,0.25);
}
.hero h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0;
}

/* ── Card ── */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #ede9fe;
}

/* ── Subject Badge ── */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.3px;
}
.badge-math    { background: #fef3c7; color: #92400e; }
.badge-science { background: #d1fae5; color: #065f46; }
.badge-english { background: #dbeafe; color: #1e40af; }
.badge-default { background: #ede9fe; color: #4c1d95; }

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.2rem auto;
    max-width: 85%;
    font-size: 0.97rem;
    line-height: 1.5;
}
.bubble-meta {
    text-align: right;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 1rem;
}
.bubble-ai {
    background: white;
    border: 1px solid #ede9fe;
    color: #1f2937;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.2rem auto 1rem 0;
    max-width: 90%;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(79,70,229,0.07);
}

/* ── Input styling ── */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #ede9fe !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    background: white !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    border: 2px solid #4f46e5 !important;
    color: #4f46e5 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 2px solid #ede9fe !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #4f46e5 !important;
}

/* ── Divider ── */
hr { border-color: #ede9fe; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #9ca3af;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.95rem; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #f3f4f6;
    border-radius: 999px;
    height: 8px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ai_tutor_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        st.error("⚠️ Model files not found. Make sure ai_tutor_model.pkl and vectorizer.pkl are present.")
        st.stop()

ml_model, vectorizer = load_model()

# ── Configure Gemini ──────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini(model_name):
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    return genai.GenerativeModel(model_name)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

# ── Helper: subject badge ─────────────────────────────────────────────────────
def subject_badge(subject):
    s = subject.lower()
    cls = "badge-math" if "math" in s else "badge-science" if "science" in s else "badge-english" if "english" in s else "badge-default"
    icon = "🔢" if "math" in s else "🔬" if "science" in s else "📖" if "english" in s else "📚"
    return f'<span class="badge {cls}">{icon} {subject}</span>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0 if st.session_state.gemini_model == "gemini-2.5-flash" else 1,
        help="Flash is faster; Pro is more thorough."
    )
    st.session_state.gemini_model = model_choice

    st.markdown("---")
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How to use**")
    st.caption("1. Type your academic question\n2. Hit **Predict** to see the subject\n3. Hit **Ask Tutor** for a full explanation")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 AI Tutor</h1>
    <p>Your smart academic helper — ask any question in Math, Science, or English</p>
</div>
""", unsafe_allow_html=True)

# ── Input Area ────────────────────────────────────────────────────────────────
with st.container():
    question = st.text_input(
        "Your question",
        placeholder="e.g., What is Newton's second law?  |  Solve 3x + 6 = 18  |  Explain a metaphor",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        predict_btn = st.button("🔍 Predict Subject", use_container_width=True, type="secondary")
    with col2:
        ask_btn = st.button("🧠 Ask Tutor", use_container_width=True, type="primary")
    with col3:
        if st.button("✕ Clear", use_container_width=True):
            st.rerun()

# ── Predict Only ──────────────────────────────────────────────────────────────
if predict_btn and question.strip():
    with st.spinner("Detecting subject..."):
        vec = vectorizer.transform([question])
        subject = ml_model.predict(vec)[0]
        confidence = ml_model.predict_proba(vec).max()

    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
            <span style="font-weight:700; color:#374151;">Subject Detected</span>
            {subject_badge(subject)}
        </div>
        <div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.3rem;">Confidence: <b>{confidence:.1%}</b></div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Ask Tutor ─────────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    with st.spinner("Your tutor is thinking..."):
        vec = vectorizer.transform([question])
        subject = ml_model.predict(vec)[0]
        confidence = ml_model.predict_proba(vec).max()

        prompt = f"""You are a friendly, encouraging academic tutor for students.

Student's question: "{question}"

The subject detected is **{subject}** (confidence: {confidence:.1%}).

Instructions:
- Answer clearly, step by step
- Use simple language a student can understand
- Use examples where helpful
- Use LaTeX for math (e.g. $2x + 5 = 15$)
- Keep a warm, encouraging tone
- End with a short motivational tip if appropriate"""

        try:
            gemini = get_gemini(st.session_state.gemini_model)
            response = gemini.generate_content(prompt)
            answer = response.text

            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "subject": subject,
                "confidence": confidence,
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as e:
            st.error(f"Gemini error: {e}")

# ── Chat History ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("---")
    st.markdown("#### 💬 Conversation")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            badge = subject_badge(msg.get("subject", ""))
            conf = msg.get("confidence", 0)
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="bubble-meta">{badge}&nbsp;&nbsp;{conf:.1%} confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="bubble-ai">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💡</div>
        <p>Ask your first question above to get started!<br>
        Math, Science, English — your AI tutor is ready.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#9ca3af; font-size:0.8rem;">AI Tutor • ML Subject Detection + Google Gemini • Made with ❤️ for education access</p>',
    unsafe_allow_html=True
)
