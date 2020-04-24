import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

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

# Base model - no hyperparameter tuning 
clf = RandomForestClassifier()
model = clf.fit(X_train_tfidf, y_train)
predicted_y = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, predicted_y)
prec = precision_score(y_test, predicted_y, average=None, zero_division=0)
recall = recall_score(y_test, predicted_y, average=None, zero_division=0)

print('Accuracy score:', acc)
print('Precision score:', prec)
print('Recall score:', recall)
print(classification_report(y_test, predicted_y))

# generate the table of precision and recall scores
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

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
k = StratifiedKFold(n_splits=2)

# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, and use all available cores

# Random search of parameters
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
grid_accuracy = evaluate(best_grid, X_test_tfidf, y_test)
print(best_grid)

########### Evaluation Metrics ###########

def print_improvement(model_accuracy):
    print('Improvement of {:0.2f}%.'.format( 100 * (model_accuracy - base_accuracy) / base_accuracy))

base_accuracy = evaluate(model, X_test_tfidf, y_test)
random_accuracy = evaluate(best_random, X_test_tfidf, y_test)
grid_accuracy = evaluate(best_grid, X_test_tfidf, y_test)
print_improvement(random_accuracy)
print_improvement(grid_accuracy)
