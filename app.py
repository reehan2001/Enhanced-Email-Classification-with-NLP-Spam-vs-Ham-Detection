from flask import Flask, render_template , request, jsonify
from nltk import NaiveBayesClassifier, classify, word_tokenize
import os
import codecs 
import random
from nltk import word_tokenize

app = Flask(__name__)

def data_load(folder):
    file_list = os.listdir(folder)
    a_list = []
    for a_file in file_list:
        if not a_file.startswith("."):
            file_path = os.path.join(folder, a_file)
            with codecs.open(file_path, 'r', encoding="ISO-8859-1", errors="ignore") as f:
                a_list.append(f.read())
    return a_list

spam_list = data_load('data/enron1/spam/')
ham_list = data_load('data/enron1/ham/')

all_emails = [(emails_list,"spam") for emails_list in spam_list]
all_emails += [(emails_list,"ham") for emails_list in ham_list]

random.seed(42)
random.shuffle(all_emails)
print(f" Spam_emails + ham_emails = Total emails Dataset size is {str(len(all_emails))} emails")

def get_features(text): 
    features = {}
    word_list = [word for word in word_tokenize(text.lower())]
    for word in word_list:
        features[word] = True
    return features

def train(features, proportion):
    train_size = int(len(features) * proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print(f"Training set size = {len(train_set)} emails")
    print(f"Test set size = {len(test_set)} emails")
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

all_features = [(get_features(email), label) for (email, label) in all_emails]
train_set, test_set, classifier = train(all_features, 0.8)


@app.route('/')
def homepage():
    return render_template('Home.html')


@app.route("/classify", methods=["POST"])
def classify_email():
    email = request.form["email"]
    prediction = classifier.classify(get_features(email))
    return render_template("result.html", prediction=prediction)



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)