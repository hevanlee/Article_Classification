import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# rf = RandomForestRegressor(random_state = 42)
from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV, validation_curve, StratifiedKFold, GridSearchCV

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

# Fit to training set
count = CountVectorizer()
X_train_count = count.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

X_test_counts = count.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

clf = RandomForestClassifier() #random_state=0
model = clf.fit(X_train_tfidf, y_train)
predicted_y = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, predicted_y)
# prec = precision_score(y_test, predicted_y, average=None, zero_division=0)
# recall = recall_score(y_test, predicted_y, average=None, zero_division=0)

print('Accuracy score:', acc)
# print('Precision score:', prec)
# print('Recall score:', recall)

# print(classification_report(y_test, predicted_y))

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, predicted_y))
# print('\n')

############################ GRID SEARCH CV #################################

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 100, 150, 200, None],
    'max_features': ['auto', 'log2', None],
    'min_samples_leaf': [1, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 300, 500, 1000]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train_tfidf, y_train)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, X_test_tfidf, y_test)
print(best_grid)

############################ RANDOM SEARCH CV ###############################

# # number of trees in random forest
# # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# # number of features at every split
# max_features = ['auto', 'sqrt']
# # max depth
# # 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, None]
# max_depth = [int(x) for x in np.linspace(100, 500, num = 11)] 
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# criterion = ['gini', 'entropy']

# # print(n_estimators)
# # print(max_depth)

# # create random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'bootstrap': bootstrap,
# }

# k = StratifiedKFold(n_splits=5)

# # Random search of parameters
# rfc_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = k, verbose=2, random_state=42, n_jobs = -1)
# # Fit the model
# rfc_random.fit(X_train_tfidf, y_train)
# # print results
# print(rfc_random.best_params_)
