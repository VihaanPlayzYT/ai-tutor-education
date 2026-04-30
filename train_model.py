import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

print("Training AI Tutor Model...\n")

# Good balanced dataset
data = {
    'question': [
        "What is the formula for acceleration due to gravity?",
        "Explain Newton's three laws of motion",
        "How does gravity work on Earth?",
        "What is kinetic energy?",
        "Solve for x in 2x + 5 = 15",
        "Solve the quadratic equation x² - 5x + 6 = 0",
        "How do I calculate the area of a circle?",
        "Explain ionic and covalent bonding",
        "What happens when acid reacts with base?",
        "What is photosynthesis?",
        "What is the difference between RAM and ROM?",
        "Write a Python program to print hello world",
        "How do I declare a variable in Python?",
        "What is a loop in programming?",
        "Why is grammar important in English?",
        "What is the difference between there and their?"
    ],
    'subject': [
        'Physics', 'Physics', 'Physics', 'Physics',
        'Math', 'Math', 'Math',
        'Chemistry', 'Chemistry', 'Chemistry',
        'Computer Science', 'Computer Science', 'Computer Science', 'Computer Science',
        'English', 'English'
    ]
}

df = pd.DataFrame(data)

X = df['question']
y = df['subject']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = SVC(kernel='linear', probability=True)
model.fit(X_vec, y)

print("Model trained on", len(df), "questions.")
print("Model is ready!\n")

# Test function
def ai_tutor(question):
    question_vec = vectorizer.transform([question])
    prediction = model.predict(question_vec)[0]
    print(f"Question: {question}")
    print(f"Predicted Subject: {prediction}")
    print("-" * 50)

# Test the tutor
print("Testing AI Tutor:")
ai_tutor("What is the formula for acceleration due to gravity?")
ai_tutor("Solve for x in 2x + 5 = 15")
ai_tutor("Explain ionic and covalent bonding")
ai_tutor("How do I declare a variable in Python?")
ai_tutor("Why is grammar important in English?")

joblib.dump(model, 'ai_tutor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\n✅ Model trained and saved successfully!")