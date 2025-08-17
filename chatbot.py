import json
import random
import pickle
from utils import tokenize, stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the intents data with error handling
try:
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: The 'intents.json' file was not found.")
    print("Please make sure the file exists in the data directory.")
    exit()
except json.JSONDecodeError:
    print("Error: The 'intents.json' file has a syntax error.")
    print("Please check the file for missing commas, brackets, or other formatting issues.")
    exit()

# --- Model Eğitim Bölümü ---
# Bu bölüm, chatbot başladığında sadece bir kez çalışacak

# Toplam kelime dağarcığını ve etiketleri toplama
all_patterns = []
all_tags = []

for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(tag)

# Metin verisini vektörleştirmek ve bir sınıflandırıcı eğitmek için Pipeline oluşturma
model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=True)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Veriyi eğitime ayırma
# Küçük bir veri seti için test_size değerini dikkatli seçmeliyiz
if len(all_tags) > 10 and len(set(all_tags)) < len(all_tags) * 0.8:
    X_train, X_test, y_train, y_test = train_test_split(all_patterns, all_tags, test_size=0.2, random_state=42, stratify=all_tags)
    model_pipeline.fit(X_train, y_train)
else:
    # Yeterli veri yoksa tüm veriyi kullanarak eğitin
    print("Warning: Not enough data for a robust train/test split. Training on all available data.")
    model_pipeline.fit(all_patterns, all_tags)

# Modeli bir dosyaya kaydetme
with open('data/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)


# --- Model Yükleme ve Kullanım Bölümü ---

def get_sentiment_score(sentence):
    """
    Analyzes the sentiment of a sentence and returns a compound score.
    Score is between -1 (most negative) and +1 (most positive).
    """
    sentiment = analyzer.polarity_scores(sentence)
    return sentiment['compound']

def get_response(input_sentence):
    # Load the trained model
    try:
        with open('data/chatbot_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        print("Model file not found or corrupted. Please run the script once to train the model.")
        return "I'm sorry, an error occurred. I cannot understand you."
    
    # Analyze sentiment of the user input
    sentiment_score = get_sentiment_score(input_sentence)

    # Check for negative sentiment
    if sentiment_score < -0.3:
        return "I'm sorry to hear that. I understand you're upset. How can I help you better?"
    
    # Predict intent using the trained model
    predicted_tag = loaded_model.predict([input_sentence])[0]
    
    # Find the corresponding response
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    
    # Fallback response if model finds no match
    return "I'm sorry, I don't understand that. Can you rephrase?"


# Start the chatbot
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = get_response(user_input)
    print(f"Bot: {response}")