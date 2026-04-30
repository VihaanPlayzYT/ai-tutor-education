# 🎓 AI Tutor for Underprivileged Students

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://ai-tutor-education-typ9qdwwxfmzaqnohzrg5v.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)

> **Making quality education accessible to every student — regardless of their background.**

---

## 🌍 The Problem

Millions of students in India lack access to personalized academic support. Without tutors or guidance, a simple unanswered question can become a barrier to learning.

As the **IT Head at my school** and a **volunteer teacher for underprivileged students**, I witnessed this gap first-hand. I built this project to do something about it.

---

## 💡 What It Does

AI Tutor is a machine learning-powered web app that:

1. **Accepts** a question from a student
2. **Classifies** it into the correct subject — Math, Science, or English
3. **Suggests** relevant learning resources and guidance instantly

No internet searches. No waiting. Just instant, accessible academic support.

---

## 🚀 Live Demo

👉 **[Try it here](https://ai-tutor-education-typ9qdwwxfmzaqnohzrg5v.streamlit.app/)**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | ML model training & classification |
| Pandas & NumPy | Data preprocessing |
| Streamlit | Web app interface & deployment |
| Google Colab | Model development environment |

---

## ⚙️ How It Works

```
Student types a question
        ↓
Text is vectorized (TF-IDF)
        ↓
Trained ML classifier predicts subject
        ↓
Relevant resources are suggested
```

The model was trained on a labeled dataset of student questions across Math, Science, and English, using a TF-IDF vectorizer paired with a scikit-learn classifier, saved as `ai_tutor_model.pkl`.

---

## 📁 Project Structure

```
ai-tutor-education/
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── ai_tutor_model.pkl    # Trained ML classifier
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🏃 Run Locally

```bash
git clone https://github.com/VihaanPlayzYT/ai-tutor-education.git
cd ai-tutor-education
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Impact

- 👨‍🏫 Personally taught **basic coding to 50+ underprivileged students**
- 🤖 This project extends that impact through AI — available 24/7, at zero cost to the student

---

## 🔮 Roadmap

- [ ] Upgrade to NLP-based chatbot for conversational tutoring
- [ ] Add support for **multiple Indian languages** (Hindi, Tamil, Telugu, etc.)
- [ ] Expand subject coverage beyond 3 categories
- [ ] Integrate curated YouTube/NCERT resource links per topic
- [ ] Mobile-friendly UI improvements

---

## 👨‍💻 About the Builder

Hi, I'm **Vihaan** — IT Head at my school and a volunteer educator passionate about using technology to bridge the education gap in India. This project is my attempt to give every student access to the kind of instant academic support that most take for granted.

---

Made with ❤️ for education access · Built by [VihaanPlayzYT](https://github.com/VihaanPlayzYT)
