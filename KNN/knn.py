import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def calculate_accuracy(y_test, y_pred):
    correct = (y_pred == y_test).sum()
    total_instances = len(y_test)
    accuracy = correct / total_instances
    return correct, total_instances, accuracy

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

def count_zeros_ones(k_nearest, distances):
    count_0 = 0
    count_1 = 0
    for element in k_nearest:
        if element == 0:
            count_0 += 1
        elif element == 1:
            count_1 += 1

    if count_0 > count_1:
        return 0
    elif count_0 < count_1:
        return 1
    else:
        distances_0 = 0
        distances_1 = 0
        for i in range(len(k_nearest)):
            if k_nearest[i] == 0:
                distances_0 += 1/distances[i]
            elif k_nearest[i] == 1:
                distances_1 += 1/distances[i]
        if distances_0 > distances_1:
            return 0
        else:
            return 1


class KNN:
    def __init__(self):
        self.k = 1

    def setK(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict(X_test[i]) for i in range(X_test.shape[0])]
        return predictions

    def _predict(self, x_test):
        distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]
        # sort and return nums of rows before sort
        rows_num = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i][0] for i in rows_num]
        distances = sorted(distances)
        predict_value = count_zeros_ones(k_nearest, distances)
        return predict_value


path = 'diabetes.csv'
diabetes = pd.read_csv(path)
print('diabetes')
print(diabetes)
print()
print("*"*70)
print()

# the features and targets are separated
num_of_cols = diabetes.shape[1]
X = diabetes.iloc[:, 0:num_of_cols-1]
y = diabetes.iloc[:, num_of_cols-1:num_of_cols]
print("features")
print(X)
print("---------------------------------------------")
print("targets")
print(y)
print()
print("*"*70)
print()

# divide data into 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=45)
print("X_train")
print(X_train)
print("---------------------------------------------")
print("X_test")
print(X_test)
print("---------------------------------------------")
print("y_train")
print(y_train)
print("---------------------------------------------")
print("y_test")
print(y_test)
print()
print("*"*70)
print()

# Normalize each feature column separately for training and test objects using Min-Max Scaling.
X_train = (X_train-X_train.min()) / (X_train.max()-X_train.min())

X_test = (X_test-X_test.min()) / (X_test.max()-X_test.min())

print("X_train after standardization")
print(X_train)
print("---------------------------------------------")
print("X_test after standardization")
print(X_test)
print()
print("*"*70)
print()

# Convert X to numpy array
X_train = X_train.to_numpy().reshape((-1, num_of_cols - 1))
X_test = X_test.to_numpy().reshape((-1, num_of_cols - 1))

# Convert y to numpy array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

knn = KNN()
knn.fit(X_train, y_train)
avg = 0
iterations = 8
for k in range(2, iterations+2):
    print('k value:', k)
    knn.setK(k)
    y_pred = knn.predict(X_test)
    y_pred = np.array(y_pred).reshape(len(y_pred), 1)
    correct, total_instances, accuracy = calculate_accuracy(y_test, y_pred)
    avg += accuracy
    print('Number of correctly classified instances:', correct)
    print('Total number of instances:', total_instances)
    print('Accuracy:', accuracy)
    print()

print("*"*70)
print()
avg = avg/iterations
print('The average accuracy across all iterations:', avg)