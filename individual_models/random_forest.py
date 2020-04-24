import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV, validation_curve, StratifiedKFold

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

clf = RandomForestClassifier()
model = clf.fit(X_train_tfidf, y_train)

predicted_y = model.predict(X_test_tfidf)

print('Accuracy score:', accuracy_score(y_test, predicted_y)) 
# print('Precision score:', precision_score(y_test, predicted_y, average=None, zero_division=0))
# print('Recall score:', recall_score(y_test, predicted_y, average=None, zero_division=0))

# generate the table of precision and recall scores
# print(classification_report(y_test, predicted_y))

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

###########

# Random Forest Classifier

# # max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 2)]
max_depth.append(None)


# # number of features at every split
# max_features = ['auto', 'sqrt']
max_features = ['auto']

# # number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 9000, num = 1000)]

# # create random grid
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
print("=======")

random_predicted_y = rfc_random.predict(X_test_tfidf)
print('After random forest tuning - Accuracy score:', accuracy_score(y_test, random_predicted_y))

#{'n_estimators': 946, 'max_features': 'auto', 'max_depth': 500}
new = {
    'n_estimators': [1258], 
    'max_features': ['auto'], 
    'max_depth': [500]
    }




new_model = RandomForestClassifier(n_estimators = 100)
# Fit the model
new_model.fit(X_train_tfidf, y_train)

base_accuracy = evaluate(new_model, X_test_tfidf, y_test)
best_random = rfc_random.best_params_
random_accuracy = evaluate(best_random, X_test_tfidf, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
##########


## grid search
grid = GridSearchCV(estimator=clf, param_grid=new, cv=k)

# grid = GridSearchCV(estimator=rfc, param_grid=rfc_random.best_params_, cv=10)

# Fit
grid.fit(X_train_tfidf, random_predicted_y)
print(grid.best_params_)
best_grid = grid.best_estimator_
grid_accuracy = evaluate(best_grid, X_test_tfidf, random_predicted_y)
print(best_grid)

# Predict
grid_search = clf.predict(X_test_tfidf)

print('Accuracy score:', accuracy_score(y_test, random_predicted_y))
print('Precision score:', precision_score(y_test, random_predicted_y, average=None, zero_division=0))
print('Recall score:', recall_score(y_test, random_predicted_y, average=None, zero_division=0))

print(classification_report(y_test, random_predicted_y))
