from sklearn.ensemble import GradientBoostingClassifier as boost
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from util import plot_learning_curve, plot_tuning_curve,after_tuned_evaluation, load_breastCancer, load_winequality
from sklearn.model_selection import train_test_split, cross_validate,ShuffleSplit

def boost_learning_rate(trainX, trainy, title):
    learning_rate_list = np.linspace(0.01, 2, 20)
    test_scores = []
    train_scores = []
    for i in learning_rate_list:
        clf = boost(learning_rate=i)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, learning_rate_list, "Boosting learning_rate","Boosting learning_rate vs accuracy of "+title)   


def boost_n_estimators(trainX, trainy, title):
    n_estimators_list = np.linspace(1, 160, 20).astype(int)
    test_scores = []
    train_scores = []
    for i in n_estimators_list:
        clf = boost(n_estimators=i)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, n_estimators_list, "Boosting num of estimators","Boosting num of estimators vs accuracy of "+title) 



def breastCancerboost():
    breastCancer_training_X, breastCancer_training_y, breastCancer_test_X,  breastCancer_test_y = load_breastCancer()

    boost_learning_rate(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    boost_n_estimators(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    clf = boost(n_estimators=45, learning_rate=1)
    after_tuned_evaluation(clf, breastCancer_training_X, breastCancer_training_y, breastCancer_test_X, breastCancer_test_y, "breast Cancer boost ")
    
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y, title="breast cancer learning curve boost")


def winequalityboost():
    winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y = load_winequality()

    boost_learning_rate(winequality_training_X, winequality_training_y, 'winequality')
    boost_n_estimators(winequality_training_X, winequality_training_y, 'winequality')
    
    clf = boost(n_estimators=45, learning_rate=1)
    after_tuned_evaluation(clf, winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "wine quality boost ")
    plot_learning_curve(clf, winequality_training_X, winequality_training_y, title="winequality learning curve boost")


if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    breastCancerboost()
    winequalityboost()