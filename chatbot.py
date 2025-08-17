import json
import random
import pickle
from utils import tokenize, stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer and user data store.
analyzer = SentimentIntensityAnalyzer()
user_data = {
    'name': None,
    'state': 'initial'
}

# Load the intents data from the JSON file.
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

# --- Model Training and Evaluation Section ---
# This section prepares the data, trains the model, and tests its performance.

all_patterns = []
all_tags = []

for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(tag)

model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=True)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split data into training and testing sets.
if len(all_tags) > 10 and len(set(all_tags)) < len(all_tags) * 0.8:
    X_train, X_test, y_train, y_test = train_test_split(all_patterns, all_tags, test_size=0.2, random_state=42, stratify=all_tags)
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model_pipeline.predict(X_test)

    # Evaluate the model's accuracy.
    print("--- Model Performance Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--- End of Report ---\n")
else:
    print("Warning: Not enough data for a robust train/test split. Training on all available data.")
    model_pipeline.fit(all_patterns, all_tags)

with open('data/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)


# --- Chatbot Logic Section ---
# This section defines the core functions for chat interaction.

def get_sentiment_score(sentence):
    sentiment = analyzer.polarity_scores(sentence)
    return sentiment['compound']

def get_response(input_sentence):
    try:
        with open('data/chatbot_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return "I'm sorry, an error occurred. I cannot understand you."
    
    sentiment_score = get_sentiment_score(input_sentence)
    if sentiment_score < -0.3:
        return "I'm sorry to hear that. I understand you're upset. How can I help you better?"

    if user_data['state'] == 'awaiting_order_number':
        if any(char.isdigit() for char in input_sentence):
            order_number = ''.join(filter(str.isdigit, input_sentence))
            user_data['state'] = 'initial'
            return f"Thank you. I'm checking the status for order number {order_number}..."
        else:
            return "That doesn't look like a valid order number. Can you please enter it again?"

    predicted_tag = loaded_model.predict([input_sentence])[0]
    
    if predicted_tag == "name_capture":
        words = input_sentence.split()
        try:
            name_index = -1
            if "my name is" in input_sentence.lower():
                name_index = words.index("is") + 1
            elif "you can call me" in input_sentence.lower():
                name_index = words.index("me") + 1
            elif "i am" in input_sentence.lower():
                name_index = words.index("am") + 1
            elif "i'm" in input_sentence.lower():
                name_index = words.index("i'm") + 1
            
            if name_index != -1 and name_index < len(words):
                user_name = " ".join(words[name_index:])
                user_data['name'] = user_name.capitalize()
                return f"Hello, {user_data['name']}! It's nice to meet you."
            else:
                return "I didn't quite catch your name. Can you please tell me again?"
        except ValueError:
            return "I didn't quite catch that. Can you please state your name clearly?"

    if predicted_tag == 'order_status':
        user_data['state'] = 'awaiting_order_number'
        return "To check your order status, I need your order number. Can you please provide it?"

    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            if predicted_tag == "greeting" and user_data['name']:
                return random.choice([f"Hello, {user_data['name']}! How can I help you today?",
                                      f"Hi, {user_data['name']}! What can I do for you?",
                                      "What's up?"])
            return random.choice(intent['responses'])
    
    return "I'm sorry, I don't understand that. Can you rephrase?"

# --- Main Chat Loop ---
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = get_response(user_input)
    print(f"Bot: {response}")