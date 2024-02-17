import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Weight:
    def __init__(self, l, m):
        self.w = np.random.rand(l, m)

class NeuralNetwork:
    def setArchitecture(self,num_input_neurons,num_hidden_neurons,num_out_neurons,num_epochs,learning_rate):
        self.num_input_neurons = num_input_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_out_neurons = num_out_neurons
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.wh = Weight(num_hidden_neurons,num_input_neurons)
        self.wo = Weight(num_out_neurons,num_hidden_neurons)

    def train(self,X_train,Y_train):
        for e in range(self.num_epochs):
            y_pred = []
            for c in range(X_train.shape[0]):
                x_train = X_train[c]
                y_train = Y_train[c]
                # Feedforward
                ah = np.sum(self.wh.w * x_train, axis=1)
                ah = sigmoid(ah)
                ao = np.sum(self.wo.w * ah, axis=1)
                y_pred.append(ao[0])
                # Backpropagation
                error_o = (ao - y_train)
                sum = [0 for _ in range(self.num_hidden_neurons)]
                for i in range(self.num_hidden_neurons):
                    for j in range(len(error_o)):
                        sum[i] += error_o[j]*self.wo.w[j][i]
                sum = np.array(sum)
                error_h = sum * ah * (1 - ah)
                # (Weight update)
                self.wo.w = self.wo.w - self.learning_rate * error_o * ah
                res = np.outer(error_h, x_train)
                self.wh.w = self.wh.w - self.learning_rate * res

            y_pred = np.array(y_pred).reshape(len(y_pred), 1)
            print('The Error in epoch', e+1, ': ', neuralNetwork.calculate_error(Y_train, y_pred))


    def predict(self,X_test):
        y_pred = []
        for c in range(X_test.shape[0]):
            x_test = X_test[c]
            ah = np.sum(self.wh.w * x_test, axis=1)
            ah = sigmoid(ah)
            ao = np.sum(self.wo.w * ah, axis=1)
            y_pred.append(ao[0])

        return y_pred

    def calculate_error(self, y_test, y_pred):
        return np.mean((y_test - y_pred)**2)


data = pd.read_excel('concrete_data.xlsx')

data = shuffle(data, random_state=35)

num_of_cols = data.shape[1]
X = data.iloc[:, 0:num_of_cols - 1]
y = data.iloc[:, num_of_cols - 1:num_of_cols]

X_train = X[0:525]
y_train = y[0:525]
X_test = X[525:]
y_test = y[525:]

mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = X_train.to_numpy().reshape((-1, 4))
X_test = X_test.to_numpy().reshape((-1, 4))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


neuralNetwork = NeuralNetwork()
neuralNetwork.setArchitecture(4, 4, 1, 1000, 0.0001)
neuralNetwork.train(X_train,y_train)
y_pred = neuralNetwork.predict(X_test)
y_pred = np.array(y_pred).reshape(len(y_pred), 1)
print('y_pred')
print(y_pred)
print('------------------------------------------------')
print('The Error: ', neuralNetwork.calculate_error(y_test, y_pred))
print('------------------------------------------------')
print('r2_score: ', r2_score(y_test, y_pred))

# enter file to predict
# print('------------------------------------------------')
# fileName = input('Enter File Name:')
# input_data = pd.read_excel(fileName)
# X_input = input_data.iloc[:, 0:4]
# X_input = (X_input - mean) / std
# X_input = X_input.to_numpy().reshape((-1, 4))
# y_pred_1 = neuralNetwork.predict(X_input)
# y_pred_1 = np.array(y_pred_1).reshape(len(y_pred_1), 1)
# print(y_pred_1)

# enter new record to predict
print('------------------------------------------------')
list_input = input("Enter new record: ").split()
n1,n2,n3,n4 = map(float, list_input)
X_input = np.array([[n1, n2, n3, n4]])
X_input = (X_input - mean.to_numpy()) / std.to_numpy()
y_pred_1 = neuralNetwork.predict(X_input)
y_pred_1 = np.array(y_pred_1).reshape(len(y_pred_1), 1)
print(y_pred_1)
