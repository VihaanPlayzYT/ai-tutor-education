import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

print("Training Improved AI Tutor Model...\n")

# Much better and larger training dataset
data = {
    'question': [
        # Physics (10 examples)
        "What is the formula for acceleration due to gravity?",
        "Explain Newton's three laws of motion",
        "How does gravity work on Earth?",
        "What is kinetic energy?",
        "What is potential energy?",
        "What is Newton's second law of motion?",
        "How do you calculate force using F=ma?",
        "What is the speed of light?",
        "Explain projectile motion",
        "What is Ohm's law?",

        # Math (10 examples)
        "Solve for x in 2x + 5 = 15",
        "Solve the quadratic equation x² - 5x + 6 = 0",
        "How do I calculate the area of a circle?",
        "What is the Pythagorean theorem?",
        "Solve 3x - 7 = 20",
        "Find the roots of x² + 4x + 4 = 0",
        "What is the circumference of a circle?",
        "What is 25% of 400?",
        "Solve the system: x + y = 10, 2x - y = 4",
        "Calculate the derivative of 3x² + 2x",

        # Chemistry (10 examples)
        "Explain ionic and covalent bonding",
        "What happens when acid reacts with base?",
        "What is photosynthesis?",
        "What is the pH scale?",
        "Explain the periodic table",
        "Balance the chemical equation H2 + O2 → H2O",
        "What is Avogadro's number?",
        "What is electrolysis?",
        "Explain types of chemical reactions",
        "What are hydrocarbons?",

        # Computer Science (10 examples)
        "Write a Python program to print hello world",
        "How do I declare a variable in Python?",
        "What is a loop in programming?",
        "What is the difference between RAM and ROM?",
        "Explain what an algorithm is",
        "How does if-else statement work in Python?",
        "What is a function in programming?",
        "Explain object-oriented programming",
        "What is HTML used for?",
        "What is binary search?",

        # English (10 examples)
        "Why is grammar important in English?",
        "What is the difference between there and their?",
        "Explain simile and metaphor",
        "What is the difference between noun and verb?",
        "Correct this sentence: He go to school yesterday",
        "What is active and passive voice?",
        "What is a synonym for beautiful?",
        "How do you write a formal email?",
        "What is the past tense of 'eat'?",
        "Explain the meaning of 'ubiquitous'"
    ],
    'subject': [
        'Physics']*10 + ['Math']*10 + ['Chemistry']*10 + ['Computer Science']*10 + ['English']*10
}

df = pd.DataFrame(data)

print(f"Training on {len(df)} questions...")

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(df['question'])

model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_vec, df['subject'])

print("Model trained successfully!")

# Test the improved model
def test_ai_tutor(question):
    question_vec = vectorizer.transform([question])
    prediction = model.predict(question_vec)[0]
    confidence = model.predict_proba(question_vec).max()
    
    print(f"Question: {question}")
    print(f"Predicted Subject: {prediction}")
    print(f"Confidence: {confidence:.1%}")
    print("-" * 60)

print("\nTesting Improved Model:")
test_ai_tutor("Solve for x in 2x + 5 = 15")
test_ai_tutor("Explain Newton's three laws of motion")
test_ai_tutor("Explain ionic and covalent bonding")
test_ai_tutor("How do I declare a variable in Python?")
test_ai_tutor("What is the difference between there and their?")

# Save the improved files
joblib.dump(model, 'ai_tutor_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\n✅ Improved model and vectorizer saved successfully!")
print("You can now upload the new ai_tutor_model.pkl and vectorizer.pkl to GitHub.")
