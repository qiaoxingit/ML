import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from util import plot_learning_curve, plot_tuning_curve, after_tuned_evaluation, load_winequality, load_breastCancer
from sklearn.model_selection import train_test_split,  GridSearchCV,learning_curve,ShuffleSplit, cross_validate
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing


# pre pruning max_depth
def tree_max_depth(trainX, trainy, title):
    max_depth = np.linspace(1, 22, 15, dtype=int)
    train_scores=[]
    test_scores=[]
    for i in max_depth:
        clf = DecisionTreeClassifier(max_depth=i)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, max_depth, "max depth","max depth vs accuracy of "+title)   



def tree_alpha(trainX, trainy, title):
    alphas = np.linspace(0.0, 0.01, 10)
    train_scores=[]
    test_scores=[]
    for i in alphas:
        clf = DecisionTreeClassifier(ccp_alpha=i)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, alphas, "ccp_alpha","ccp_alpha vs accuracy of "+title)   


# post pruning cost_complexity


def complexity_alpha(trainX, trainy, testX, testy, title):
    clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(trainX, trainy)
    ccp_alphas = path.ccp_alphas
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(trainX, trainy)
        clfs.append(clf)

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    train_scores = [clf.score(trainX, trainy) for clf in clfs]
    test_scores = [clf.score(testX, testy) for clf in clfs]
    

    fig, ax = plt.subplots()
    # plt.xlim(0.0, 0.003)
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("decision tree tune alpha " +title)
    ax.plot(ccp_alphas, train_scores, marker='o',
            label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o',
            label="test", drawstyle="steps-post")
    ax.legend()
    plt.grid()
    plt.savefig('images/' + "DT " + title +" complexity tune for decision tree")
    plt.clf()


def breastCancerDT():
    

    breastCancer_training_X, breastCancer_training_y, breastCancer_test_X,  breastCancer_test_y = load_breastCancer()

    tree_max_depth(breastCancer_training_X, breastCancer_training_y, 'breastCancer')

    tree_alpha(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    complexity_alpha(breastCancer_training_X, breastCancer_training_y,
        breastCancer_test_X, breastCancer_test_y, 'breastCancer')

    clf = DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=1, ccp_alpha=0.004)

    after_tuned_evaluation(clf, breastCancer_training_X, breastCancer_training_y, breastCancer_test_X, breastCancer_test_y, "breast cancer DT ")
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y,
                    title="breast cancer learning curve decision tree")

def winequalityDT():
    
    winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y = load_winequality()

    tree_max_depth(winequality_training_X, winequality_training_y, 'winequality')

    tree_alpha(winequality_training_X, winequality_training_y,
        'winequality') 
    complexity_alpha(winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "winequality")

    clf = DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=1, ccp_alpha=0.000)

    after_tuned_evaluation(clf, winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "wine quality DT ")
    plot_learning_curve(clf, winequality_training_X, winequality_training_y,
                    title="wine quality learning curve decision tree")


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    # breastCancerDT()
    winequalityDT()