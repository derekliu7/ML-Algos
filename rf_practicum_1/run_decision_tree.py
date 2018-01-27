import pandas as pd
from DL_DecisionTree import DecisionTree
from DL_RandomForest import RandomForest


def test_tree(filename):
    df = pd.read_csv(filename)
    y = df.pop('Result').values
    X = df.values

    # tree = DecisionTree()
    # tree.fit(X, y)
    # print(tree)
    # y_predict = tree.predict()

    forest = RandomForest(10, 3)
    forest.fit(X, y)
    forest.predict(X)
    print(forest.score(X, y))
    # print('%26s   %10s   %10s' % ("FEATURES", "ACTUAL", "PREDICTED"))
    # print('%26s   %10s   %10s' % ("----------", "----------", "----------"))
    # for features, true, predicted in zip(X, y, y_predict):
    #     print('%26s   %10s   %10s' %
    #           (str(features), str(true), str(predicted)))


if __name__ == '__main__':
    test_tree(
        '/Users/DL/Desktop/Galvanize_Work_Product/DSCI6003-student/week4/4.2/data/playgolf.csv')
