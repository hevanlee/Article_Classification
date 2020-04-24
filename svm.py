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

all_labels = {
    'ARTS CULTURE ENTERTAINMENT': 1,
    'BIOGRAPHIES PERSONALITIES PEOPLE': 2,
    'DEFENCE': 3,
    'DOMESTIC MARKETS': 4,
    'FOREX MARKETS': 5,
    'HEALTH': 6,
    'MONEY MARKETS': 7,
    'SCIENCE AND TECHNOLOGY': 8,
    'SHARE LISTINGS': 9,
    'SPORTS': 10,
    'IRRELEVANT': 0
}
for topic in topics:
    labels.append(all_labels[topic])

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
probas = clf.predict_proba(X_test_tfidf)

# Store each article, classification and probability
class_proba = []

for i in range(0, len(probas)):
    article = {}
    article['article_id'] = i
    article['features'] = X_test[i]
    article['class'] = list(all_labels.keys())[list(all_labels.values()).index(y_pred[i])] 
    article['probability'] = max(probas[i])
    class_proba.append(article)

# Uncomment below to see
# print('First article:', class_proba[0])

print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Precision score:', precision_score(y_test, y_pred, average=None, zero_division=0))
print('Recall score:', recall_score(y_test, y_pred, average=None, zero_division=0))

print(classification_report(y_test, y_pred))

# print("Best parameters:")
# best_params = clf.best_estimator_.get_params()
# for param in sorted(best_params.keys()):
#     print("\t%s: %r" % (param, best_params[param]))
