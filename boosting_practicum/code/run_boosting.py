from DL_boosting import AdaBoostBinaryClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    data = np.genfromtxt(
        '/Users/DL/Desktop/Galvanize_Work_Product/DSCI6003-student/week5/5.2/boosting_practicum/data/spam.csv', delimiter=',')

    y = data[:, -1]
    x = data[:, 0:-1]

    train_x, test_x, train_y, test_y = train_test_split(x, y)

    my_ada = AdaBoostBinaryClassifier(n_estimators=50)
    my_ada.fit(train_x, train_y)
    # print(my_ada.predict(test_x))
    print("Accurracy of my adaboost:", my_ada.score(test_x, test_y))

    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(train_x, train_y)
    # print(ada.predict(test_x))
    print("Accuracy of sklearn:", ada.score(test_x, test_y))
