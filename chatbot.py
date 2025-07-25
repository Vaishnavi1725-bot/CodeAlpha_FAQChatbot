import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load FAQs from file
with open('faqs.json', 'r') as file:
    faqs = json.load(file)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# Text preprocessing
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

cleaned_questions = [clean_text(q) for q in questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

# Chatbot
print("\n🤖 Welcome to the FAQ Chatbot!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    cleaned_input = clean_text(user_input)
    user_vector = vectorizer.transform([cleaned_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    max_sim_index = similarity.argmax()
    max_score = similarity[0][max_sim_index]

    if max_score > 0.3:
        print("Bot:", answers[max_sim_index])
    else:
        print("Bot: Sorry, I don’t understand that question.")