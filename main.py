import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np 
import tflearn
import tensorflow
import random
import json  
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words ,labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs = []
    docs_tag = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word_token = nltk.word_tokenize(pattern)
            words.extend(word_token)
            docs.append(word_token)
            docs_tag.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs):
        bag = []
        
        word_token = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in word_token:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_tag[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words ,labels, training, output),f)

# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try: 
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=5,batch_size=8)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for token in s_words:
        for i, word in enumerate(words):
            if word == token:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("start talking with the bot (type quit to stop)!")
    while True:
        command = input("You: ")
        if command.lower() == "quit":
            break

        results = model.predict([bag_of_words(command,words)])[0]
        results_index = np.argmax(results)

        if results[results_index]>0.7:

            tag = labels[results_index]
            for tags in data["intents"]:
                if tags["tag"] == tag:
                    response = tags["responses"]
                    print(response)
        else:
            print("I didnt get that, try again")

chat()
