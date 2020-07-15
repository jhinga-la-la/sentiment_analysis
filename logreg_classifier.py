import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

reviews_test = []
reviews_train = []

for line in open('data/aclImdb/movie_data/full_train.txt', encoding="utf8"):
    reviews_train.append(line.strip())

for line in open('data/aclImdb/movie_data/full_test.txt', encoding="utf8"):
    reviews_test.append(line.strip())

print("An example from train set of review:\n{}\n".format(reviews_train[0]))

# Learn regex, very important for manipulating string in NLP applications

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
    reviews = [REPLACE_WITH_SPACE.sub(" ", REPLACE_NO_SPACE.sub("", review.lower())) for review in reviews]
    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)
print('---------------------------------------------------------\n')
print("An example from train set of review after processing:\n{}\n".format(reviews_train[0]))

# Convert a collection of text documents to a matrix of token counts
# CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

# Based on the dataset structure, first 12.5K belongs to +1 class and remaining 12.5K belongs to -1 class
target = [1 if x < 12500 else 0 for x in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size=0.75)

# modify hyper-parameter C, which adjusts the regularization
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c, max_iter=200)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.05, max_iter=200)
final_model.fit(X, target)
print("\nFinal Accuracy: %s\n" % accuracy_score(target, final_model.predict(X_test)))
# Final Accuracy: 0.88128

feature_to_coef = {word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}

for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(best_positive)

print('\n---------------------------------------------------------\n')

for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
    print(best_negative)
