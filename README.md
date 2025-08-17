# Simple Chatbot with NLP

This is a basic rule-based chatbot created with Python and the NLTK library. It uses Natural Language Processing (NLP) techniques to match user input to predefined patterns and provide appropriate responses.

## Features

-   **Rule-based:** The chatbot's responses are based on patterns defined in a JSON file.
-   **NLP-enabled:** Uses NLTK for tokenization and stemming to handle variations in user phrases.
-   **Extensible:** Easily add new conversation intents by updating the `intents.json` file.

## Requirements

-   Python 3.6+

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Download the necessary NLTK data:
    ```bash
    python -c "import nltk; nltk.download('punkt') and nltk.download('punkt_tab')"
    ```

## Usage

1.  Run the main script:
    ```bash
    python chatbot.py
    ```
2.  Start chatting with the bot! Type `quit` to exit the conversation.

## Project Structure

Basic_ChatBot/
├── chatbot.py
├── utils.py
├── requirements.txt
├── README.md
└── data/
    └── intents.json

## How It Works

The chatbot uses a simple pattern-matching approach:
1.  User input is tokenized (split into words).
2.  Words are stemmed (reduced to their root form).
3.  The stemmed words are compared against the patterns in `intents.json`.
4.  If a match is found, a random response from the corresponding intent is selected and displayed.