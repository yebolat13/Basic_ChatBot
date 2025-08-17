import json
import random
from utils import tokenize, stem

# Load the intents data from the JSON file with error handling
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: The 'intents.json' file was not found.")
    print("Please make sure the file exists in the same directory.")
    exit()
except json.JSONDecodeError:
    print("Error: The 'intents.json' file has a syntax error.")
    print("Please check the file for missing commas, brackets, or other formatting issues.")
    exit()

# Collect all patterns and tags
all_words = []
all_tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    all_tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and remove duplicates from the list of all words
all_words = [stem(w) for w in all_words if w.isalnum()]
all_words = sorted(list(set(all_words)))
all_tags = sorted(list(set(all_tags)))

def find_best_match(input_sentence):
    """
    Finds the best matching intent based on a simple word count.
    """
    tokenized_input = tokenize(input_sentence)
    max_match = 0
    best_match_tag = None
    
    for intent in intents['intents']:
        match_count = 0
        for pattern in intent['patterns']:
            pattern_stems = [stem(w) for w in tokenize(pattern)]
            
            input_stems = [stem(w) for w in tokenized_input]
            
            for w in input_stems:
                if w in pattern_stems:
                    match_count += 1
            
        if match_count > max_match:
            max_match = match_count
            best_match_tag = intent['tag']
            
    return best_match_tag, max_match

# Main chat loop
def get_response(input_sentence):
    best_match_tag, max_match = find_best_match(input_sentence)
    
    if max_match > 0 and best_match_tag:
        for intent in intents['intents']:
            if intent['tag'] == best_match_tag:
                return random.choice(intent['responses'])
    
    # Fallback response if no match is found
    return "I'm sorry, I don't understand that. Can you rephrase?"

# Start the chatbot
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = get_response(user_input)
    print(f"Bot: {response}")