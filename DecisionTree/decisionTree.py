import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import random

path = 'drug.csv'
data = pd.read_csv(path)
print('data')
print(data)
print()
print("*"*70)
print()

missing_values = data.isnull().sum()
print("Missing values in each column in data:")
print(missing_values)
print()
print("*"*70)
print()

data = data.dropna()
print('data after removing missing values')
print(data)
print()
print("*"*70)
print()

def categorize_features(data):
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

    return categorical_features, numerical_features


categorical, numerical = categorize_features(data)

print("Categorical Features:", categorical)
print()
print("*"*70)
print()

# the features and targets are separated
num_of_cols = data.shape[1]
X = data.iloc[:, 0:num_of_cols - 1]
y = data.iloc[:, num_of_cols - 1:num_of_cols]
print("features")
print(X)
print("---------------------------------------------")
print("targets")
print(y)
print()
print("*"*70)
print()

# First experiment
print('First experiment')
print('-----------------')
# Generate a list of 5 unique random numbers
random_numbers = random.sample(range(1, 101), 5)
print('random_numbers: ', random_numbers)
print()
highest = [0,0,0]
count = 1
for random_seed in random_numbers:
    # the data is shuffled and split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=random_seed)

    # print("X_train")
    # print(X_train)
    # print("---------------------------------------------")
    # print("X_test")
    # print(X_test)
    # print("---------------------------------------------")
    # print("y_train")
    # print(y_train)
    # print("---------------------------------------------")
    # print("y_test")
    # print(y_test)
    # print()
    # print("*"*70)
    # print()

    # categorical features are encoded
    categorical_columns_X = categorical[:-1]
    label_encoder_X = LabelEncoder()

    for i in range(len(categorical_columns_X)):
        X_train[categorical_columns_X[i]] = label_encoder_X.fit_transform(X_train[categorical_columns_X[i]])
        X_test[categorical_columns_X[i]] = label_encoder_X.transform(X_test[categorical_columns_X[i]])

    # print("X_train after encoding")
    # print(X_train)
    # print("---------------------------------------------")
    # print("X_test after encoding")
    # print(X_test)
    # print("---------------------------------------------")
    # print()
    # print("*" * 70)
    # print()

    # categorical targets are encoded
    categorical_columns_y = categorical[-1:]
    label_encoder_y = LabelEncoder()
    y_train[categorical_columns_y[0]] = label_encoder_y.fit_transform(y_train[categorical_columns_y[0]])
    y_test[categorical_columns_y[0]] = label_encoder_y.transform(y_test[categorical_columns_y[0]])

    # print("y_train after encoding")
    # print(y_train)
    # print("---------------------------------------------")
    # print("y_test after encoding")
    # print(y_test)
    #
    # print()
    # print("*" * 70)
    # print()

    maxD = random.randint(2, 4)
    model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=maxD)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Experiment #'+str(count)+':')
    accuracy = model.score(X_test, y_test)
    print("Accuracy: ", accuracy)
    print("Tree depth:", model.tree_.max_depth)
    print("Total number of nodes:", model.tree_.node_count)
    count+=1
    if highest[0] < accuracy:
        highest = [accuracy, model.tree_.max_depth, model.tree_.node_count]
    elif highest[0] == accuracy:
        if highest[1] < model.tree_.max_depth:
            highest = [accuracy, model.tree_.max_depth, model.tree_.node_count]

    print()
    print("-" * 50)
    print()

print('The highest overall performance')
print("Accuracy: ", highest[0])
print("Tree depth:", highest[1])
print("Total number of nodes:", highest[2])

print()
print("*"*70)
print()

# -----------------------------------------------------------------------------------------------------------------------

# Second experiment
print('Second experiment')
print('-----------------')

# categorical features are encoded
categorical_columns_X = categorical[:-1]
label_encoder_X = LabelEncoder()

for i in range(len(categorical_columns_X)):
    X[categorical_columns_X[i]] = label_encoder_X.fit_transform(X[categorical_columns_X[i]])

print("X after encoding")
print(X)
print()
print("*" * 70)
print()

# categorical targets are encoded
categorical_columns_y = categorical[-1:]
label_encoder_y = LabelEncoder()
y[categorical_columns_y[0]] = label_encoder_y.fit_transform(y[categorical_columns_y[0]])

print("y after encoding")
print(y)
print()
print("*" * 70)
print()

trainSizes = [30, 40, 50, 60, 70]
means_accuracy = []
means_nodes = []
for trainSize in trainSizes:
    print('train size:', trainSize)
    # Generate a list of 5 unique random numbers
    random_numbers = random.sample(range(1, 101), 5)
    print('random_numbers: ', random_numbers)
    print()
    accuracies = []
    sizes = []
    for random_seed in random_numbers:
        # the data is shuffled and split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=(trainSize/100), random_state=random_seed)
        # print("X_train")
        # print(X_train)
        # print("---------------------------------------------")
        # print("X_test")
        # print(X_test)
        # print("---------------------------------------------")
        # print("y_train")
        # print(y_train)
        # print("---------------------------------------------")
        # print("y_test")
        # print(y_test)
        # print()
        # print("*"*70)
        # print()

        maxD = random.randint(2, 4)
        model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=maxD)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        size = model.tree_.node_count
        accuracies.append(accuracy)
        sizes.append(size)

    print('Mean, Maximum, and Minimum accuracy')
    print('------------------------------------')
    mean_acc = sum(accuracies)/len(accuracies)
    maximum_acc = max(accuracies)
    minimum_acc = min(accuracies)
    print('Mean: ', mean_acc)
    print('Maximum: ', maximum_acc)
    print('Minimum: ', minimum_acc)
    print()
    means_accuracy.append(mean_acc)
    print('Mean, Maximum, and Minimum tree size')
    print('------------------------------------')
    mean_node = sum(sizes)/len(sizes)
    maximum_node = max(sizes)
    minimum_node = min(sizes)
    print('Mean: ', mean_node)
    print('Maximum: ', maximum_node)
    print('Minimum: ', minimum_node)
    means_nodes.append(mean_node)
    print()
    print('*'*70)
    print()

plt.plot(trainSizes,means_accuracy, marker='o', linestyle='-', color='b', label='Line Plot')
plt.xlabel('Train Sizes')
plt.ylabel('Mean Accuracies')
plt.title('Accuracy against Training Set Size')
plt.legend()
plt.show()

plt.plot(trainSizes,means_nodes, marker='o', linestyle='-', color='b', label='Line Plot')
plt.xlabel('Train Sizes')
plt.ylabel('Number of Nodes')
plt.title('Number of Nodes in Final Tree against Training Set Size')
plt.legend()
plt.show()
