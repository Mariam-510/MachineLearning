import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
from joblib import load

# load data
data_pro = pd.read_csv('sentimentdataset (Project 1).csv')
# print the first rows
print(data_pro.head())
print()

# print describtion
print(data_pro.describe())
print()

# distribution of samples in each class
distribution = data_pro['Target'].value_counts()
print(distribution)
print()

# plot distribution of samples in each class
plt.figure(figsize=(8, 6))
sns.countplot(x='Target', data=data_pro)
plt.title('Distribution of Samples in Each Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# drop id and source columns
data = data_pro.drop(columns=['Source', 'ID'])
print(data)
print()

# spacy.cli.download("en_core_web_sm")
lemmatization = spacy.load('en_core_web_sm')

# eliminate stop words, perform lemmatization for each sentiment
def perform_lemmatization(all_messages):
    all_messages = lemmatization(all_messages)
    new_sentiment = [sentiment.lemma_ for sentiment in all_messages if not sentiment.is_stop]
    return " ".join(new_sentiment)

data['Message'] = data['Message'].apply(perform_lemmatization)

print(data)
print()

# Generate sentence embeddings using CountVectorizer
count_vectorizer = CountVectorizer()
embeddings = count_vectorizer.fit_transform(data['Message'])
print(embeddings.toarray())
print()

# Split into training and testing sets (75 train, 25 test)
X = embeddings
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print("X_train")
print(X_train.toarray())
print("---------------------------------------------")
print("X_test")
print(X_test.toarray())
print("---------------------------------------------")
print("y_train")
print(y_train)
print("---------------------------------------------")
print("y_test")
print(y_test)

# Initial Experiment
# use LinearSVC
linear_svc = LinearSVC(dual=False)
# use grid search to identify optimal parameters from this values [0.01,0.1, 1, 10]
# cross validation is 3
gridSearchCV = GridSearchCV(linear_svc, {'C': [0.01,0.1,0.5,1]}, cv=3, scoring='accuracy', verbose=1)
gridSearchCV.fit(X_train, y_train)

# Optimal Parameter
print("Optimal Parameter: ", gridSearchCV.best_params_)

# predict test data
y_pred = gridSearchCV.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Highest Accuracy: {:.2f}%".format(accuracy * 100))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# classification report for initial experiment
print("Classification Report for Initial Experiment:")
print(classification_report(y_test, y_pred))

# Save the best model of SVC
bestModel = gridSearchCV.best_estimator_
dump(bestModel, 'bestModel.joblib')

# reload
loadedModel = load('bestModel.joblib')

new_data = pd.read_csv('newdata.csv')
print(new_data)
print()

new_data['Message'] = new_data['Message'].apply(perform_lemmatization)
print(new_data)
print()

embeddings1 = count_vectorizer.transform(new_data['Message'])
# predict
y_pred = loadedModel.predict(embeddings1)
print('y_pred',y_pred)


# Subsequent Experiment
highestAccuracy = 0
optimalHyperparameters = {}

# Hyperparameter
num_neurons = [64, 128, 256]
num_learning_rate = [0.001, 0.01, 0.1]
num_batch_size = [32, 64, 128]

# Explore different hyperparameters
for neurons in num_neurons:
    for learningRate in num_learning_rate:
        for batchSize in num_batch_size:

            # ANN model
            model = Sequential()
            model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            optimizer = Adam(learning_rate=learningRate)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # fit the model
            history = model.fit(X_train.toarray(), y_train, epochs=20, batch_size=batchSize,
                                validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

            # calculate accuracy
            loss, accuracy = model.evaluate(X_test.toarray(), y_test)
            # predict test data
            # probabilities values
            y_p = model.predict(X_test.toarray())
            # convert probabilities values to 0 or 1
            y_pred = [1 if p > 0.5 else 0 for p in y_p]
            # classification report
            classificationReport = classification_report(y_test, y_pred)

            # compare accuracies to find the highest accuracy
            if accuracy > highestAccuracy:
                highestAccuracy = accuracy
                # optimal neuron, learning rate, batch size
                optimalHyperparameters = {'neurons': neurons, 'learningRate': learningRate, 'batchSize': batchSize}
                # optimal classification report
                optimalClassificationReport = classificationReport
                # Save the best model
                model.save('bestModelANN.keras')

print()
print("Optimal Hyperparameters:", optimalHyperparameters)
print("Highest Accuracy: {:.2f}%".format(highestAccuracy * 100))
print("Classification Report for Best Model:")
print(optimalClassificationReport)

# Reload
loadedModelANN = load_model('bestModelANN.keras')

new_data = pd.read_csv('newdata.csv')
print(new_data)
print()

new_data['Message'] = new_data['Message'].apply(perform_lemmatization)
print(new_data)
print()

embeddings1 = count_vectorizer.transform(new_data['Message'])

y_p = loadedModelANN.predict(embeddings1.toarray())
# convert probabilities values to 0 or 1
y_pred = [1 if p > 0.5 else 0 for p in y_p]

print('y_pred',y_pred)
print()