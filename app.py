from flask import Flask, render_template, request
from src.chatbot import get_response  # Import the chatbot logic

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Renders the web page

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["message"]  # Get the message from the user input
    response = get_response(user_message)   # Get response from the chatbot
    return response

if __name__ == "__main__":
    app.run(debug=True)  # Running the app in debug mode
