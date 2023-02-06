import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from util import plot_learning_curve, plot_tuning_curve
from sklearn.model_selection import train_test_split,  GridSearchCV,learning_curve,ShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score


# pre pruning max_depth
def tree_max_depth(trainX, trainy, title):
    max_depth = range(1, 30, 2)
    train_scores=[]
    test_scores=[]
    for i in max_depth:
        clf = DecisionTreeClassifier(max_depth=i, random_state=0)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, max_depth, "max depth","max depth vs accuracy of "+title)   


# post pruning cost_complexity


def alpha(trainX, trainy, testX, testy, title):
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(trainX, trainy)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(trainX, trainy)
        clfs.append(clf)

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    train_scores = [clf.score(trainX, trainy) for clf in clfs]
    test_scores = [clf.score(testX, testy) for clf in clfs]
    

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("decision tree tune alpha " +title)
    ax.plot(ccp_alphas, train_scores, marker='o',
            label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o',
            label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig('images/' + title+" alpha tune for decision tree")
    plt.clf()


def breastCancerDT():
    breastCancer = pd.read_csv('breastCancer.csv', sep=',')

    X = breastCancer.values[:, 0:-1]
    y = breastCancer.values[:, -1]

    breastCancer_training_X, breastCancer_test_X, breastCancer_training_y, breastCancer_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    tree_max_depth(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    alpha(breastCancer_training_X, breastCancer_training_y,
        breastCancer_test_X, breastCancer_test_y, 'breastCancer')

    clf = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=3, random_state=0)
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y,
                    title="breast cancer learning curve decision tree")

def winequalityDT():
    winequality = pd.read_csv('winequality.csv', sep=',')

    X = winequality.values[:, 0:-1]
    y = winequality.values[:, -1]

    winequality_training_X, winequality_test_X, winequality_training_y, winequality_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    tree_max_depth(winequality_training_X, winequality_training_y, 'winequality')

    alpha(winequality_training_X, winequality_training_y,
        winequality_test_X, winequality_test_y, 'winequality') 

    clf = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, random_state=0)
    plot_learning_curve(clf, winequality_training_X, winequality_training_y,
                    title="wine quality learning curve decision tree")


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    breastCancerDT()
    winequalityDT()