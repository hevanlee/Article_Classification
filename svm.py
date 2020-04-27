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
class_proba_1 = []
class_proba_2 = []
class_proba_3 = []
class_proba_4 = []
class_proba_5 = []
class_proba_6 = []
class_proba_7 = []
class_proba_8 = []
class_proba_9 = []
class_proba_10 = []
class_proba_0= []

class_proba = []

# Sort through predictions, matching article numbers, features, topic, and probability
for i in range(0, len(probas)):
    article = {}
    article['article_number'] = i + 9501
    article['article_words'] = X_test[i]
    article['topic'] = list(all_labels.keys())[list(all_labels.values()).index(y_pred[i])] 
    article['probability'] = max(probas[i])
    article['label'] = y_test[i]
    
    if y_pred[i] == 1:
        class_proba_1.append(article)
    elif y_pred[i] == 2:
        class_proba_2.append(article)
    elif y_pred[i] == 3:
        class_proba_3.append(article)
    elif y_pred[i] == 4:
        class_proba_4.append(article)
    elif y_pred[i] == 5:
        class_proba_5.append(article)
    elif y_pred[i] == 6:
        class_proba_6.append(article)
    elif y_pred[i] == 7:
        class_proba_7.append(article)
    elif y_pred[i] == 8:
        class_proba_8.append(article)
    elif y_pred[i] == 9:
        class_proba_9.append(article)
    elif y_pred[i] == 10:
        class_proba_10.append(article)
    elif y_pred[i] == 0:
        class_proba_0.append(article)
    
    class_proba.append(article)

# Check X_test and class_proba articles are in line
for i in range(0, len(class_proba)):
    if class_proba[i]['article_words'] != X_test[i]:
        print("X_test and class_proba not lined up")
        exit()

# Expected distribution of articles in test set by topic
expected_num = {
    'ARTS CULTURE ENTERTAINMENT': 6,
    'BIOGRAPHIES PERSONALITIES PEOPLE': 9,
    'DEFENCE': 14,
    'DOMESTIC MARKETS': 7,
    'FOREX MARKETS': 44,
    'HEALTH': 10,
    'MONEY MARKETS': 88,
    'SCIENCE AND TECHNOLOGY': 4,
    'SHARE LISTINGS': 11,
    'SPORTS': 10,
    'IRRELEVANT': 249
}

# Recommend up to expected num of articles above benchmark
recommendations = []

class_proba = []
class_proba.append(sorted(class_proba_1, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_2, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_3, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_4, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_5, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_6, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_7, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_8, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_9, key = lambda i: i['probability'], reverse=True))
class_proba.append(sorted(class_proba_10, key = lambda i: i['probability'], reverse=True))

for topic in expected_num.keys():
    recommended = {}
    article_nums = []
    i = 0
    if topic == 'IRRELEVANT':
        break
    while i < expected_num[topic] and i < len(class_proba[all_labels[topic] - 1]) and i < 10:
        if class_proba[all_labels[topic] - 1][i]['probability'] > 0.2:
            article_nums.append(class_proba[all_labels[topic] - 1][i])
            i += 1
    
    article_nums = sorted(article_nums, key = lambda i: i['article_number'])
    recommended[topic] = article_nums
    recommendations.append(recommended)

# Recommendations
print("Recommended articles:")
for i in range(0, len(recommendations)):
    for topic in recommendations[i].keys():
        nums = []
        for article in recommendations[i][topic]:
            nums.append(article['article_number'])
        print(topic + ":", nums)
print('__________________________________________________________________________________\n')

# Count number of articles per y_test category)
# list(all_labels.keys())[list(all_labels.values()).index(value)] --> given value 'value', finds corresponding key in all_labels{}
# apologies to whoever is marking this for code towards the end

total_per_topic = {}
for label in y_test:
    if list(all_labels.keys())[list(all_labels.values()).index(label)] not in total_per_topic.keys():
        total_per_topic[list(all_labels.keys())[list(all_labels.values()).index(label)]] = 1
    else:
        total_per_topic[list(all_labels.keys())[list(all_labels.values()).index(label)]] += 1

# Calculate metrics for recommendations
print("Metrics for recommendations:")

for i in range(0, len(recommendations)):
    for topic in recommendations[i].keys():
        tp = 0
        total_rec = 0
        total_topic = 0
        for article in recommendations[i][topic]:
            if article['topic'] == list(all_labels.keys())[list(all_labels.values()).index(article['label'])]:
                tp += 1
            total_rec += 1
    
        if topic in total_per_topic.keys():
            total_topic = total_per_topic[topic]
    
        print(topic + ":")
        if total_rec == 0:
            print("Precision: 0.00, 0 articles correctly recommended out of 0 total recommended")
        else:
            precision = tp/total_rec
            print("Precision: %1.2f, %d articles correctly recommended out of %d total recommended" %(precision, tp, total_rec))

        if total_topic == 0:
            print("Recall: 0.0, %d articles correctly recommended out of 0 total articles in category" %(tp))
        else:
            recall = tp/total_topic
            print("Recall: %1.2f, %d articles correctly recommended out of %d total articles in category" %(recall, tp, total_topic))
        
        
        if precision + recall == 0:
            print("F1: 0.00")
        else:
            f1 = 2*((precision * recall)/(precision + recall))
            print("F1: %1.2f" %(f1))
        print()
print('__________________________________________________________________________________\n')

# Comparing model and predictions with actual labels
print("Metrics for model test data:")
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Precision score:', precision_score(y_test, y_pred, average=None, zero_division=0))
print('Recall score:', recall_score(y_test, y_pred, average=None, zero_division=0))

print(classification_report(y_test, y_pred))