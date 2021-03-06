import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("training.csv")

data = df["article_words"]
topics = df["topic"]

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

label_number = [0,1,2,3,4,5,6,7,8,9,10]

# Create target vector
y = labels

# Divide training set into training and development sets
X_train = data[:9000]
X_test = data[9000:]
y_train = y[:9000]
y_test = y[9000:]

# Fit to training set
count = CountVectorizer()
X_train_count = count.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

X_test_counts = count.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

names = ["K Nearest Neighbours", "Bernoulli Naive Bayes",
"Multinomial Naive Bayes", "Linear SVM",
"Decision Tree Classifier", "Random Forest Classifier", 
"AdaBoost Classifier", "Multiple Layer Perceptron"]

classifiers = [
    KNeighborsClassifier(),
    BernoulliNB(alpha=0),
    MultinomialNB(alpha=0),
    SGDClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(
        hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.1, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10
        # Used shell script here to auto-run every parameter instead since hypertuning takes too long to run

    ) # for ADAM 73.8% when run w/ alpha = 1, 78.2% when run w/ alpha = 0.1, 76% w/ alpha = 0.5, 77.6% when run w/ alpha = 0.01
      # for LBFGS, accuracy was highest between alpha 0.2 to 0.4
]

model_scores = {}
counter = 0
for name, clf in zip(names, classifiers):
    if (name == "Multiple Layer Perceptron"):
        model = clf.fit(X_train_tfidf, y_train)
        predicted_y = model.predict(X_test_tfidf)
        print(classification_report(y_test, predicted_y))
        print(accuracy_score(y_test,predicted_y))
