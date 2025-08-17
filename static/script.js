function sendMessage() {
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const userMessage = userInput.value;

    if (userMessage.trim() === "") {
        return;
    }

    // Display user message
    const userDiv = document.createElement("div");
    userDiv.classList.add("message", "user-message");
    userDiv.innerHTML = userMessage;
    chatBox.appendChild(userDiv);

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send message to the server
    fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
        .then(response => response.json())
        .then(data => {
            const botMessage = data.response;
            const botDiv = document.createElement("div");
            botDiv.classList.add("message", "bot-message");
            botDiv.innerHTML = botMessage;
            chatBox.appendChild(botDiv);

            // Scroll to the bottom again
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            console.error('Error fetching bot response:', error);
            const errorDiv = document.createElement("div");
            errorDiv.classList.add("message", "bot-message");
            errorDiv.innerHTML = "Sorry, something went wrong. Please try again.";
            chatBox.appendChild(errorDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        });

    // Clear the input field
    userInput.value = "";
}

// Enable sending messages with the "Enter" key
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});