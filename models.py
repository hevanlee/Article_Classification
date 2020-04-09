import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

# Create target vector
y = labels

# Divide training set into training (9000) and development (500) sets
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
    BernoulliNB(),
    MultinomialNB(),
    SGDClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier(alpha=0.1) #73.8% when run w/ alpha = 1, 78.2% when run w/ alpha = 0.1, 76% w/ alpha = 0.5, 77.6% when run w/ alpha = 0.01
]

model_scores = {}

for name, clf in zip(names, classifiers):
    print("--" + name)
    model = clf.fit(X_train_tfidf, y_train)
    predicted_y = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, predicted_y)
    prec = precision_score(y_test, predicted_y, average=None, zero_division=0)
    recall = recall_score(y_test, predicted_y, average=None, zero_division=0)

    print('Accuracy score:', acc)
    print('Precision score:', prec)
    print('Recall score:', recall)

    print(classification_report(y_test, predicted_y))
    model_scores[name] = acc

model_view = [(v,k) for k,v in model_scores.items()]
model_view.sort(reverse=True)
print("--Models ranked in descending order--")
for v,k in model_view:
    print("%(model)s: %(acc).1f" %{'model': k, 'acc': v * 100.0})

