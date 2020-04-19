import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("../training.csv")

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

# pipeline combining text feature extractor with SGDClassifier
pipeline = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    SGDClassifier()
)

hyperparameters = {
    'sgdclassifier__loss': ['hinge'],#, 'squared_hinge'],
    'sgdclassifier__alpha': [0.0001 ],#, 0.00001, 0.001, 0.01, 0.1, 1],
    'sgdclassifier__max_iter': [5000] #, 1000, 10000]
}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Precision score:', precision_score(y_test, y_pred, average=None, zero_division=0))
print('Recall score:', recall_score(y_test, y_pred, average=None, zero_division=0))

print(classification_report(y_test, y_pred))

# print("Best parameters:")
# best_params = clf.best_estimator_.get_params()
# for param in sorted(best_params.keys()):
#     print("\t%s: %r" % (param, best_params[param]))