# Basic ChatBot

![Chatbot Demo](https://via.placeholder.com/600x400?text=Basic+Chatbot+Demo)

A simple yet powerful chatbot built with Python, utilizing Natural Language Processing (NLP) to understand user intents and manage conversation flow. The project is deployed as a web application using the Flask framework.

## ✨ Features

- **Natural Language Understanding (NLU):** Uses a machine learning model to classify user messages into predefined intents (e.g., `greeting`, `order_status`, `payment`).
- **Sentiment Analysis:** Analyzes user input to detect negative emotions and provide empathetic responses.
- **Conversation Flow Management:** Handles multi-step conversations (e.g., asking for an order number after a user asks about their order status).
- **User-Specific Memory:** Remembers the user's name throughout the conversation to provide a personalized experience.
- **Web Interface:** Deployed as a web application using Flask, allowing interaction through a modern and user-friendly web UI.

## 🚀 Getting Started

Follow these steps to set up and run the chatbot on your local machine.

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
    cd [Your-Repository-Name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create this file. See the section below.)*

### Running the Application

1.  **Run the main application:**
    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to:
    `http://127.0.0.1:5000`

    You should now see the chatbot interface ready to chat!

## 🤖 How It Works

The chatbot uses a `TfidfVectorizer` and a `LogisticRegression` model to predict the user's intent. The model is trained on a JSON file containing various user patterns and their corresponding intent tags.

The `app.py` script serves as the main entry point, handling web requests, processing user input, and managing the conversation's state.

## 📁 File Structure

Basic_ChatBot/
├── data/
│   ├── intents.json            # Contains all intents, patterns, and responses
│   └── chatbot_model.pkl       # The trained machine learning model
├── static/
│   ├── script.js               # Frontend logic for the chat interface
│   └── style.css               # Styling for the web application
├── templates/
│   └── index.html              # The main HTML file for the chat UI
├── venv/                       # Virtual environment
├── .gitignore
├── app.py                      # The Flask web application
├── chatbot.py                  # The original console-based chatbot (for reference)
├── requirements.txt            # List of required Python libraries
└── README.md                   # This README file


## 📝 To-Do & Future Improvements

- **Expand Data:** Add more patterns to the `intents.json` file to improve model accuracy for all intents.
- **User Authentication:** Implement a user login system to personalize conversations across sessions.
- **Database Integration:** Connect the chatbot to a database to store and retrieve conversation history or other user data.
- **Advanced NLP:** Explore more complex models (e.g., deep learning) for enhanced intent recognition.

---