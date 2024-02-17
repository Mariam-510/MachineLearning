import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# (a)----------------------------------------------------------------------------
path = 'loan_old.csv'
data_old = pd.read_csv(path)
print('data_old')
print(data_old)
path1 = 'loan_new.csv'
data_new = pd.read_csv(path1)
print('data_new')
print(data_new)

# (b)(i)----------------------------------------------------------------------------
#  check whether there are missing values for loan_old
missing_values_old = data_old.isnull().sum()
print("Missing values in each column in loan_old:")
print(missing_values_old)
print()
print("*"*70)
print()

# (b)(ii)----------------------------------------------------------------------------
def check_type(column):
    if column.dtype == 'object':
        return 'categorical'
    elif pd.api.types.is_numeric_dtype(column):
        return 'numerical'

# Apply check_type to each column in loan_old
types_old = data_old.apply(check_type)
categorical_columns_old = types_old[types_old == 'categorical'].index.tolist()
numerical_columns_old = types_old[types_old == 'numerical'].index.tolist()

print("Categorical columns for loan_old:", categorical_columns_old)
print("Numerical columns for loan_old:", numerical_columns_old)

# Apply check_type to each column in loan_new
types_new = data_new.apply(check_type)
categorical_columns_new = types_new[types_new == 'categorical'].index.tolist()
numerical_columns_new = types_new[types_new == 'numerical'].index.tolist()

print()
print("*"*70)
print()

# (b)(iii)----------------------------------------------------------------------------
# check whether numerical features have the same scale
for i in range(len(numerical_columns_old)-1):
    column_summary = data_old[numerical_columns_old[i]].describe()
    print("Summary statistics of", numerical_columns_old[i])
    print(column_summary)
    print()

print("*"*70)
print()

# (b)(iv)----------------------------------------------------------------------------
# Visualize a pairplot between numercial columns for loan_old
pairplot_numerical_old = data_old[numerical_columns_old]
sns.pairplot(pairplot_numerical_old)
plt.show()

# (c)(i)----------------------------------------------------------------------------
# Records containing missing values are removed
data_old = data_old.dropna()
print('data_old after removing missing values')
print(data_old)
data_new = data_new.dropna()
print('data_new after removing missing values')
print(data_new)
data_new_copy = data_new.copy()

print()
print("*"*70)
print()

# (c)(ii)----------------------------------------------------------------------------
# the features and targets are separated
num_of_cols = data_old.shape[1]
X_old = data_old.iloc[:, 1:num_of_cols - 2]
y_old = data_old.iloc[:, num_of_cols - 2:num_of_cols]
print("features")
print(X_old)
print("---------------------------------------------")
print("targets")
print(y_old)

print()
print("*"*70)
print()

# (c)(iii)----------------------------------------------------------------------------
# the data is shuffled and split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_old, y_old, shuffle=True, test_size=0.25, random_state=45)
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

# (c)(iv)----------------------------------------------------------------------------
# categorical features are encoded
categorical_columns_X = categorical_columns_old[:-1]
label_encoder_X = LabelEncoder()

for i in range(1, len(categorical_columns_X)):
    X_train[categorical_columns_X[i]] = label_encoder_X.fit_transform(X_train[categorical_columns_X[i]])
    X_test[categorical_columns_X[i]] = label_encoder_X.transform(X_test[categorical_columns_X[i]])
    data_new[categorical_columns_X[i]] = label_encoder_X.transform(data_new[categorical_columns_X[i]])

print("X_train after encoding")
print(X_train)
print("---------------------------------------------")
print("X_test after encoding")
print(X_test)
print("---------------------------------------------")
print("loan_new after encoding")
print(data_new)

print()
print("*"*70)
print()

# (c)(v)----------------------------------------------------------------------------
# categorical targets are encoded
categorical_columns_y_old = categorical_columns_old[-1:]
label_encoder_y = LabelEncoder()

for i in range(len(categorical_columns_y_old)):
    y_train[categorical_columns_y_old[i]] = label_encoder_y.fit_transform(y_train[categorical_columns_y_old[i]])
    y_test[categorical_columns_y_old[i]] = label_encoder_y.transform(y_test[categorical_columns_y_old[i]])

print("y_train after encoding")
print(y_train)
print("---------------------------------------------")
print("y_test after encoding")
print(y_test)

print()
print("*"*70)
print()

# (c)(vi)----------------------------------------------------------------------------
# numerical features are standardized
numerical_columns_X = numerical_columns_old[:-2]

for feature in numerical_columns_X:
    mean_value = X_train[feature].mean()
    std_value = X_train[feature].std()
    X_train[feature] = (X_train[feature] - mean_value) / std_value
    X_test[feature] = (X_test[feature] - mean_value) / std_value
    data_new[feature] = (data_new[feature] - mean_value) / std_value

print("X_train after standardization")
print(X_train)
print("---------------------------------------------")
print("X_test after standardization")
print(X_test)
print("---------------------------------------------")
print("loan_new after standardization")
print(data_new)

print()
print("*"*70)
print()

# ----------------------------------------------------------------------------
# Convert to numpy array
X_train = X_train.to_numpy().reshape((-1, num_of_cols - 3))
X_test = X_test.to_numpy().reshape((-1, num_of_cols - 3))

# y_train for linear
y_train_lin = y_train.iloc[:, :1]
y_test_lin = y_test.iloc[:, :1]
y_train_lin = y_train_lin.to_numpy()
y_test_lin = y_test_lin.to_numpy()

# y_train for logistic
y_train_log = y_train.iloc[:, 1:]
y_test_log = y_test.iloc[:, 1:]
y_train_log = y_train_log.to_numpy()
y_test_log = y_test_log.to_numpy()

# (d)----------------------------------------------------------------------------
# Fit a linear regression model to the data to predict the loan amount
print('Linear Regression')
model = LinearRegression()
model.fit(X_train, y_train_lin)
print('Coefficients: \n', model.coef_, " ", model.intercept_)
y_pred = model.predict(X_test)

# (e)----------------------------------------------------------------------------
# Evaluate the linear regression model using sklearn's R^2 score
r2 = r2_score(y_test_lin, y_pred)
print(f"R² Score: {r2}")
print()
print("*"*70)
print()

# (f)----------------------------------------------------------------------------
# Fit a logistic regression model to the data to predict the loan status
print('Logistic Regression')

# add 1s
X_train = np.hstack((np.ones((len(X_train), 1)), X_train))
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))
theta = np.zeros((X_train.shape[1], 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return J

def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y.reshape(-1, 1)))
    return grad

def gradient_descent(X, y, theta, alpha, num_iters):
    costs = []
    for _ in range(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        cost_val = cost(theta, X, y)
        costs.append(cost_val)

    return theta, costs

alpha = 0.01
num_iters = 3000

theta, cost_history = gradient_descent(X_train, y_train_log, theta, alpha, num_iters)
print('theta',theta)

y_pred_num = sigmoid(np.dot(X_test, theta))
y_pred_log = np.round(y_pred_num)

# (g)----------------------------------------------------------------------------
# Write a function (from scratch) to calculate the accuracy of the model
def calculate_accuracy(y_test_log, y_pred_log):
    correct_predictions = (y_pred_log == y_test_log).sum()
    total_samples = len(y_test_log)
    accuracy = correct_predictions / total_samples
    return accuracy

print('Accuracy:',calculate_accuracy(y_test_log,y_pred_log))
print()
print("*"*70)
print()


# (j)----------------------------------------------------------------------------
#  predict the loan amounts
X_data_new = data_new.iloc[:, 1:]
X_data_new = X_data_new.to_numpy().reshape((-1, num_of_cols - 3))
loan_amounts_pred = model.predict(X_data_new)
print("loan_amounts_pred:")
print(loan_amounts_pred)
print()
print("*"*70)
print()
loan_amounts_pred = loan_amounts_pred.flatten().tolist()
# add Max Loan Amount column to excel sheet
loan_amounts_column = 'Max_Loan_Amount'
data_new_copy['Max_Loan_Amount'] = loan_amounts_pred

#  predict the status
X_data_new = np.hstack((np.ones((len(X_data_new), 1)), X_data_new))
status_pred_num = sigmoid(np.dot(X_data_new, theta))
status_pred_log = np.round(status_pred_num)
status_pred_log = status_pred_log.flatten().tolist()
status_pred_log = [int(x) for x in status_pred_log]
decoded_status_pred_log = label_encoder_y.inverse_transform(status_pred_log)
print("decoded_status_pred_log:")
print(decoded_status_pred_log)
print()
print("*"*70)
print()
# add Loan Status column to excel sheet
data_new_copy['Loan_Status'] = decoded_status_pred_log
data_new_copy.to_csv('loan_new1.csv', index=False)
print("Predicted Date is written to loan_new1.csv")
print("loan_new1 data")
print(data_new_copy)