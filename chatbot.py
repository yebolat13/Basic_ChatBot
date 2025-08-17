import json
import random
import pickle
from utils import tokenize, stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
# TfidfVectorizer, kelime frekanslarını ve önemini hesaba katarak daha iyi sonuç verir
model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=True)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Veriyi eğitime ayırma
X_train, X_test, y_train, y_test = train_test_split(all_patterns, all_tags, test_size=0.2, random_state=42, stratify=all_tags)

# Modeli eğitin
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Modeli bir dosyaya kaydetme
with open('data/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# --- Model Yükleme ve Kullanım Bölümü ---
# Bu bölüm, model zaten eğitilmişse çalışacak

def get_response(input_sentence):
    # Eğer model eğitilmemişse veya yüklenemezse fallback mekanizması
    try:
        with open('data/chatbot_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        print("Model file not found or corrupted. Please run the script once to train the model.")
        return "I'm sorry, an error occurred. I cannot understand you."

    # Kullanıcı girdisini model ile tahmin etme
    predicted_tag = loaded_model.predict([input_sentence])[0]
    
    # Güvenilirlik skorunu kontrol etme (isteğe bağlı, daha iyi bir tahmin için)
    # prediction_proba = loaded_model.predict_proba([input_sentence])
    # max_proba = max(prediction_proba[0])

    # Tahmin edilen etikete göre yanıt bulma
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    
    # Eğer model bir etiket bulamazsa fallback
    return "I'm sorry, I don't understand that. Can you rephrase?"


# Chatbot'u başlat
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = get_response(user_input)
    print(f"Bot: {response}")