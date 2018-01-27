import pandas as pd
from DL_ImprovedDecisionTree import ImprovedDecisionTree
from RandomForest import RandomForest
from sklearn.cross_validation import train_test_split


def test_tree(filename):
    df = pd.read_csv(filename)
    y = df.pop('Churn?').values
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tree = ImprovedDecisionTree()
    tree.fit(X_train, y_train)
    print(tree)
    tree.predict(X_test)

    # forest = RandomForest(10, 3)
    # forest.fit(X, y)
    # forest.predict(X)
    # print(forest.score(X, y))
    # print('%26s   %10s   %10s' % ("FEATURES", "ACTUAL", "PREDICTED"))
    # print('%26s   %10s   %10s' % ("----------", "----------", "----------"))
    # for features, true, predicted in zip(X, y, y_predict):
    #     print('%26s   %10s   %10s' %
    #           (str(features), str(true), str(predicted)))


if __name__ == '__main__':
    test_tree(
        '/Users/DL/Desktop/Galvanize_Work_Product/DSCI6003-student/week4/4.3/rf_practicum_2/churn.csv')
