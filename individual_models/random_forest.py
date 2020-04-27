import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

df = pd.read_csv("../training.csv")

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

# Base model - no hyperparameter tuning 
clf = RandomForestClassifier()
model = clf.fit(X_train_tfidf, y_train)
predicted_y = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, predicted_y)
prec = precision_score(y_test, predicted_y, average=None, zero_division=0)
recall = recall_score(y_test, predicted_y, average=None, zero_division=0)

print('Base Model')
print('Accuracy score:', acc)
print('Precision score:', prec)
print('Recall score:', recall)
print(classification_report(y_test, predicted_y))

########### Random Search ###########

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 2)]
max_depth.append(None)

# number of features at every split
max_features = ['auto']

# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 9000, num = 1000)]

# create random grid
random_grid = {
    'max_depth': max_depth,
    'max_features': max_features,
    'n_estimators': n_estimators
}
#cross validating
k = StratifiedKFold(n_splits=3)

# Random search of parameters, using 3 fold cross validation
rfc_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rfc_random.fit(X_train_tfidf, y_train)
print(rfc_random.best_params_)
best_random = rfc_random.best_estimator_ 

########### Grid Search ###########

param_grid = {
    'max_depth': [250, 750, 1000, None],
    'min_samples_leaf': [1, 6, 10],
    'min_samples_split': [5, 15, 20],
    'n_estimators': [50, 100, 400, 1500]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train_tfidf, y_train)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
print(best_grid)

########### Evaluation ###########

# With hyperparameter tuning 
print('Tuned Model - Grid Search')
clf = best_grid
model = clf.fit(X_train_tfidf, y_train)
predicted_y = model.predict(X_test_tfidf)
print(classification_report(y_test, predicted_y))

print('Tuned Model - Random Search')
clf = best_random
model = clf.fit(X_train_tfidf, y_train)
predicted_y = model.predict(X_test_tfidf)
print(classification_report(y_test, predicted_y))
