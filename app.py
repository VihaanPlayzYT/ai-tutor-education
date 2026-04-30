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
    background: #0f0f1a;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    padding: 1.5rem 2rem 1.2rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(79,70,229,0.25);
}
.hero h1 {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.2rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Card ── */
.card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    border: 1px solid #2d2d4e;
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
.badge-math    { background: #3d2e00; color: #fcd34d; }
.badge-science { background: #003d2e; color: #6ee7b7; }
.badge-english { background: #00213d; color: #93c5fd; }
.badge-default { background: #1e1b4b; color: #c4b5fd; }

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
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.2rem auto 1rem 0;
    max-width: 90%;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ── Input styling ── */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #2d2d4e !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    background: #1a1a2e !important;
    color: #e2e8f0 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
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
    background: #1a1a2e !important;
    border: 2px solid #6366f1 !important;
    color: #a5b4fc !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 2px solid #2d2d4e !important;
    font-family: 'Nunito', sans-serif !important;
    background: #1a1a2e !important;
    color: #e2e8f0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #4f46e5 !important;
}

/* ── Divider ── */
hr { border-color: #2d2d4e; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #a5b4fc;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 10px rgba(165,180,252,0.5)); }
.empty-state p { font-size: 0.97rem; color: #c4b5fd; line-height: 1.7; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #2d2d4e;
    border-radius: 999px;
    height: 8px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
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
    st.markdown("---")
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How to use**")
    st.caption("1. Pick your Gemini model\n2. Type your academic question\n3. Hit **Predict** to see the subject\n4. Hit **Ask Tutor** for a full explanation")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 AI Tutor</h1>
    <p>Your smart academic helper — ask any question in Math, Science, or English</p>
</div>
""", unsafe_allow_html=True)

# ── Model Selector ───────────────────────────────────────────────────────────
MODEL_INFO = {
    "gemini-2.5-flash": {
        "icon": "⚡",
        "label": "Gemini 2.5 Flash",
        "tagline": "Fast & efficient",
        "desc": "Best for quick answers, homework help, and everyday questions. Responds in seconds.",
        "color": "#fbbf24",
    },
    "gemini-2.5-pro": {
        "icon": "🧠",
        "label": "Gemini 2.5 Pro",
        "tagline": "Deep & thorough",
        "desc": "Best for complex problems, detailed explanations, and multi-step reasoning.",
        "color": "#a78bfa",
    },
}

st.markdown("<p style='font-size:0.85rem; color:#9ca3af; margin-bottom:0.4rem;'>🤖 Choose your AI model</p>", unsafe_allow_html=True)
col_flash, col_pro = st.columns(2)

for col, model_key in zip([col_flash, col_pro], MODEL_INFO.keys()):
    info = MODEL_INFO[model_key]
    selected = st.session_state.gemini_model == model_key
    border = f"2px solid {info['color']}" if selected else "2px solid #2d2d4e"
    bg = "#1a1a2e" if selected else "#13132a"
    check = "✓ " if selected else ""
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border:{border}; border-radius:12px; padding:0.65rem 0.9rem; margin-bottom:0.3rem;">
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="font-size:1.1rem;">{info['icon']}</span>
                <span style="font-weight:800; color:{info['color']}; font-size:0.88rem;">{check}{info['label']}</span>
            </div>
            <div style="font-size:0.75rem; color:#9ca3af; margin:0.15rem 0; font-style:italic;">{info['tagline']}</div>
            <div style="font-size:0.76rem; color:#c4b5fd; line-height:1.4;">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Select {info['label']}", key=f"sel_{model_key}", use_container_width=True,
                     type="primary" if selected else "secondary"):
            st.session_state.gemini_model = model_key
            st.rerun()

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
            <span style="font-weight:700; color:#c4b5fd;">Subject Detected</span>
            {subject_badge(subject)}
        </div>
        <div style="font-size:0.85rem; color:#a5b4fc; margin-bottom:0.3rem;">Confidence: <b>{confidence:.1%}</b></div>
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
    '<p style="text-align:center; color:#6366f1; font-size:0.8rem;">AI Tutor • ML Subject Detection + Google Gemini • Made with ❤️ for education access</p>',
    unsafe_allow_html=True
)
