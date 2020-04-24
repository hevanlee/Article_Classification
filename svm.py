import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv("training.csv")
test_df = pd.read_csv("test.csv")

train_length = len(df)

df = df.append(test_df)

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
# y = labels

# Resplit data into training set into training and testing sets
X_train = data[:train_length]
y_train = labels[:train_length]

X_test = data[train_length:]
y_test = labels[train_length:]

# Fit to training set
count = CountVectorizer()
X_train_count = count.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

X_test_counts = count.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

base_model = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=5000)
clf = CalibratedClassifierCV(base_model)

# Fit
clf.fit(X_train_tfidf, y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)

print('Probabilities of classification:', clf.predict_proba(X_test_tfidf))

print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Precision score:', precision_score(y_test, y_pred, average=None, zero_division=0))
print('Recall score:', recall_score(y_test, y_pred, average=None, zero_division=0))

print(classification_report(y_test, y_pred))

# print("Best parameters:")
# best_params = clf.best_estimator_.get_params()
# for param in sorted(best_params.keys()):
#     print("\t%s: %r" % (param, best_params[param]))
