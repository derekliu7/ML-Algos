import numpy as np
from sklearn.model_selection import train_test_split
import regularized_logistic_regression_stub as rlrs
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from roc_curve import roc_curve


data = np.genfromtxt(
    '/Users/DL/Desktop/Galvanize_Work_Product/DSCI6003-student/week2/2.3/logistic_regression_practicum/data/grad.csv', delimiter=',')

y = data[1:, 0]
X = data[1:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = rlrs.LogisticRegression()
new_x = model.fit(X, y)
print(sum(model.predict(new_x) == y))
print(model.coeffs)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
# probabilities = model.predict_proba(X_test)[:, 1]
#
# tpr, fpr, thresholds = roc_curve(probabilities, y_test)
#
# plt.plot(fpr, tpr)
# plt.xlabel("False Positive Rate (1 - Specificity)")
# plt.ylabel("True Positive Rate (Sensitivity, Recall)")
# plt.title("ROC plot of fake data")
# plt.show()
