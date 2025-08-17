import json
import random
import pickle
from flask import Flask, render_template, request, jsonify
from utils import tokenize, stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app and sentiment analyzer
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
user_data = {
    'name': None,
    'state': 'initial'
}

# Load the intents data with error handling
try:
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: The 'intents.json' file was not found.")
    exit()
except json.JSONDecodeError:
    print("Error: The 'intents.json' file has a syntax error.")
    exit()

# Load the trained model
try:
    with open('data/chatbot_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except (FileNotFoundError, pickle.UnpicklingError):
    print("Model file not found or corrupted. Please run the training script first.")
    exit()

def get_sentiment_score(sentence):
    sentiment = analyzer.polarity_scores(sentence)
    return sentiment['compound']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    
    # Check for negative sentiment
    sentiment_score = get_sentiment_score(user_input)
    if sentiment_score < -0.3:
        return jsonify({"response": "I'm sorry to hear that. I understand you're upset. How can I help you better?"})
        
    # Check for conversation state for multi-step tasks.
    if user_data['state'] == 'awaiting_order_number':
        if any(char.isdigit() for char in user_input):
            order_number = ''.join(filter(str.isdigit, user_input))
            user_data['state'] = 'initial'
            return jsonify({"response": f"Thank you. I'm checking the status for order number {order_number}..."})
        else:
            return jsonify({"response": "That doesn't look like a valid order number. Can you please enter it again?"})

    # Predict the user's intent
    predicted_tag = loaded_model.predict([user_input])[0]
    
    # Handle the "name" intent to save the user's name
    if predicted_tag == "name":
        if "my name is" in user_input.lower() or "i am" in user_input.lower():
            words = user_input.split()
            try:
                if "my name is" in user_input.lower():
                    name_index = words.index("is") + 1
                elif "i am" in user_input.lower():
                    name_index = words.index("am") + 1
                else:
                    name_index = -1
                
                if name_index != -1 and name_index < len(words):
                    user_name = " ".join(words[name_index:])
                    user_data['name'] = user_name.capitalize()
                    return jsonify({"response": f"Hello, {user_data['name']}! It's nice to meet you."})
            except ValueError:
                pass
        
    if predicted_tag == 'order_status':
        user_data['state'] = 'awaiting_order_number'
        return jsonify({"response": "To check your order status, I need your order number. Can you please provide it?"})
        
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            if predicted_tag == "greeting" and user_data['name']:
                response_text = random.choice([f"Hello, {user_data['name']}! How can I help you today?",
                                              f"Hi, {user_data['name']}! What can I do for you?",
                                              "What's up?"])
                return jsonify({"response": response_text})
            
            response_text = random.choice(intent['responses'])
            return jsonify({"response": response_text})
            
    return jsonify({"response": "I'm sorry, I don't understand that. Can you rephrase?"})

if __name__ == "__main__":
    app.run(debug=True)