import numpy as np
from keras.models import load_model
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
import threading
import time
import pickle

lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('chatbot_model.h5')
intents_json = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize to-do list and reminders
to_do_list = []
reminders = []

# Function to add task to the to-do list
def add_task(task):
    to_do_list.append(task)
    print(f"Task '{task}' added to your to-do list.")

# Function to view to-do list
def view_tasks():
    if not to_do_list:
        print("Your to-do list is empty.")
    else:
        print("Here are your tasks:")
        for i, task in enumerate(to_do_list, 1):
            print(f"{i}. {task}")

# Function to delete a task
def delete_task(task_number):
    if task_number - 1 < len(to_do_list):
        removed_task = to_do_list.pop(task_number - 1)
        print(f"Task '{removed_task}' removed from your to-do list.")
    else:
        print("Invalid task number.")

# Function to add reminder
def set_reminder(task, reminder_time):
    reminders.append((task, reminder_time))
    print(f"Reminder set for '{task}' at {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to check and notify reminders
def reminder_notifier():
    while True:
        current_time = datetime.now()
        for task, reminder_time in reminders[:]:
            if reminder_time <= current_time:
                print(f"Reminder: It's time for your task '{task}'!")
                reminders.remove((task, reminder_time))
        time.sleep(10)

# Start the reminder thread
reminder_thread = threading.Thread(target=reminder_notifier, daemon=True)
reminder_thread.start()

# Preprocess the user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into a bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict the intent
def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Get response based on intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses']), tag

# Handle intents and perform actions
def chatbot_response(msg):
    intents = predict_class(msg)
    response, tag = get_response(intents, intents_json)

    if tag == 'add_task':
        task = input("What task would you like to add? ")
        add_task(task)
    elif tag == 'view_tasks':
        view_tasks()
    elif tag == 'delete_task':
        task_number = int(input("Which task number would you like to delete? "))
        delete_task(task_number)
    elif tag == 'set_reminder':
        task = input("Which task would you like to set a reminder for? ")
        reminder_time_str = input("When do you want to be reminded? (e.g., YYYY-MM-DD HH:MM:SS) ")
        reminder_time = datetime.strptime(reminder_time_str, '%Y-%m-%d %H:%M:%S')
        set_reminder(task, reminder_time)
    else:
        print(response)


# Main loop to take user input and respond
while True:
    message = input("You: ")
    chatbot_response(message)
