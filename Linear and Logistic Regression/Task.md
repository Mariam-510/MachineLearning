Python program in which includes following:
a) Load the "loan_old.csv" dataset.
b) Perform analysis on the dataset to:
 i) check whether there are missing values
 ii) check the type of each feature (categorical or numerical)
 iii) check whether numerical features have the same scale
 iv) visualize a pairplot between numercial columns
c) Preprocess the data such that:
 i) records containing missing values are removed
 ii) the features and targets are separated
 iii) the data is shuffled and split into training and testing sets
 iv) categorical features are encoded
 v) categorical targets are encoded
 vi) numerical features are standardized
d) Fit a linear regression model to the data to predict the loan amount.-> Use sklearn's linear regression.
e) Evaluate the linear regression model using sklearn's R2 score.
f) Fit a logistic regression model to the data to predict the loan status.-> Implement logistic regression from scratch using gradient descent.
g) Write a function (from scratch) to calculate the accuracy of the model.
h) Load the "loan_new.csv" dataset.
i) Perform the same preprocessing on it (except shuffling and splitting).
j) Use your models on this data to predict the loan amounts and status.