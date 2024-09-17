from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import json

lemmatizer = WordNetLemmatizer()
model = load_model('models/chatbot_model.h5')

# Load data
intents = json.load(open('data/intents.json'))
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def get_response(msg):
    p = bow(msg, words)
    prediction = model.predict(np.array([p]))[0]
    max_index = np.argmax(prediction)
    tag = classes[max_index]
    for intent in intents['intents']:
        if tag == intent['tag']:
            response = np.random.choice(intent['responses'])
            return response

if __name__ == "__main__":
    while True:
        message = input("You: ")
        response = get_response(message)
        print(f"Bot: {response}")
