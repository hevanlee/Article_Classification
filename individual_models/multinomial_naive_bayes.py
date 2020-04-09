# from https://chrisalbon.com/machine_learning/naive_bayes/multinomial_naive_bayes_classifier/
# Load libraries
import numpy as np
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv('../training.csv')

data = df['article_words']
topics = df['topic']

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(data)

# Create feature matrix
X = bag_of_words

labels = []
for topic in topics:
    if topic == 'ARTS CULTURE ENTERTAINMENT':
        labels.append(1)
    elif topic == 'BIOGRAPHIES PERSONALITIES PEOPLE':
        labels.append(2)
    elif topic == 'DEFENCE':
        labels.append(3)
    elif topic == 'DOMESTIC MARKETS':
        labels.append(4)
    elif topic == 'FOREX MARKETS':
        labels.append(5)
    elif topic == 'HEALTH':
        labels.append(6)
    elif topic == 'MONEY MARKETS':
        labels.append(7)
    elif topic == 'SCIENCE AND TECHNOLOGY':
        labels.append(8)
    elif topic == 'SHARE LISTINGS':
        labels.append(9)
    elif topic == 'SPORTS':
        labels.append(10)
    else: 
        labels.append(0)

df['labels'] = labels 

# Create target vector
y = labels

# Divide training set into training (9000) and development (500) sets
X_train = X[:9000]
X_test = X[9000:]
y_train = y[:9000]
y_test = y[9000:]

# TODO: Create new unseen test instances
# test1 = count.transform([''])
# test2 = count.transform([''])

print("----bnb")
clf = BernoulliNB()
model = clf.fit(X_train, y_train)

# print(model.predict(test1))
# print(model.predict(test2))
predicted_y = model.predict(X_test)
# print(y_test, predicted_y)
# print(model.predict_proba(X_test))
print('Accuracy score:',accuracy_score(y_test, predicted_y))
print('Precision score:',precision_score(y_test, predicted_y, average=None, zero_division=0))
print('Recall score:',recall_score(y_test, predicted_y, average=None, zero_division=0))
# print(f1_score(y_test, predicted_y, average='micro'))
# print(f1_score(y_test, predicted_y, average='macro'))
print(classification_report(y_test, predicted_y))

# Train model
print("----mnb")
clf = MultinomialNB()
model = clf.fit(X_train, y_train)

# print(model.predict(test1))
# print(model.predict(test2))
predicted_y = model.predict(X_test)
# print(y_test, predicted_y)
# print(model.predict_proba(X_test))
print('Accuracy score:',accuracy_score(y_test, predicted_y))
print('Precision score:',precision_score(y_test, predicted_y, average=None, zero_division=0))
print('Recall score',recall_score(y_test, predicted_y, average=None, zero_division=0))
# print(f1_score(y_test, predicted_y, average='micro'))
# print(f1_score(y_test, predicted_y, average='macro'))
print(classification_report(y_test, predicted_y))
