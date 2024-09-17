# preprocessing.py

import json
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
intents_file = "data/intents.json"
lemmatizer = WordNetLemmatizer()

def preprocess_data(intents_file):
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.']
    
    with open(f'data/{intents_file}') as file:
        intents = json.load(file)
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents in the corpus
            documents.append((word_list, intent['tag'], intent['context']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # Lemmatize and lower each word, remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    
    # Sort classes
    classes = sorted(list(set(classes)))
    
    # Create training data
    training = []
    output_empty = [0] * len(classes)
    
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row, doc[2]])
    
    # Shuffle and convert to numpy array
    np.random.shuffle(training)
    training = np.array(training, dtype=object)
    
    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])
    contexts = np.array([item[2] for item in training])
    
    # Save data
    pickle.dump(words, open('data/words.pkl', 'wb'))
    pickle.dump(classes, open('data/classes.pkl', 'wb'))
    pickle.dump(contexts, open('data/contexts.pkl', 'wb'))
    
    return train_x, train_y


