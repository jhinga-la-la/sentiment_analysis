# Naive Bayes classification for sentiment analysis

import random

import nltk
from nltk.corpus import movie_reviews

# form a document with all reviews and their categories(class labels)
print("Starting with pre-processing and feature extraction")
documents = [(list(movie_reviews.words(file_id)), category)
             for category in movie_reviews.categories()
             for file_id in movie_reviews.fileids(category)]
random.shuffle(documents)

# find the frequency of all words in complete documents and take top 2000 words
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]


# extract feature vector for a document.(Feature for each word is indication whether the document contains that word)
def document_features(document):
    document_word = set(document)  # set of words in a document for faster computation
    features = {}  # size = (2000,)
    for word in word_features:
        features['contains({})'.format(word)] = word in document_word  # check if each word in word_features is present in a document
    return features


# Train Naive Bayes classifier
feature_sets = [(document_features(d), c) for (d, c) in documents]
print("Feature extraction for documents done")
train_set, test_set = feature_sets[100:], feature_sets[:100]

print("Training classifier")
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print("Classifier accuracy: {}\n".format(nltk.classify.accuracy(classifier, test_set)))

# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(5)
