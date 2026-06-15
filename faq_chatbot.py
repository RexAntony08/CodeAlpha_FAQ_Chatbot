import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load FAQs
with open("faqs.json", "r") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Text preprocessing
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in string.punctuation and word not in stop_words
    ]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

print("=" * 50)
print("🤖 Welcome to the FAQ Chatbot!")
print("Type 'exit' to quit.")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Thank you! Have a great day. 👋")
        break

    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.2:
        print("Bot:", answers[best_match])
    else:
        print("Bot: Sorry, I couldn't find a relevant answer for that question.")