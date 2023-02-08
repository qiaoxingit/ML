from sklearn.svm import SVC, LinearSVC
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from util import plot_learning_curve, plot_tuning_curve,after_tuned_evaluation
from sklearn.model_selection import train_test_split, cross_validate,ShuffleSplit

def svm_c(trainX, trainy, title):
    c = np.linspace(0.1, 10, 15)
    test_scores = []
    train_scores = []
    for i in c:
        clf = SVC(C = i, kernel = "rbf", gamma=0.7)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, c, "c","c vs accuracy of "+title)   


def svm_kernel(trainX, trainy, title):
    kernel_func = ['linear','poly','rbf','sigmoid']

    test_scores = []
    train_scores = []
    for i in kernel_func:
        if (i == 'poly'):
            clf = LinearSVC(C = 1, max_iter=10000)
        else:
            clf = SVC(kernel=i, C = 1)

        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    sys.stdout = open(title+ "kernel.txt", "w")
    print(kernel_func)
    print("average training score for each algorithm")
    print(np.mean(train_scores, axis = 1))
    print("============================================")
    print("average vakidation score for each algorithm")
    print(np.mean(test_scores, axis = 1) )
    
    sys.stdout.close()


def breastCancerSVM():
    breastCancer = pd.read_csv('breastCancer.csv', sep=',')

    X = breastCancer.values[:, 0:-1]
    y = breastCancer.values[:, -1]

    breastCancer_training_X, breastCancer_test_X, breastCancer_training_y, breastCancer_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    svm_c(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    svm_kernel(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    clf = SVC(C=1, kernel='rbf')
    after_tuned_evaluation(clf, breastCancer_training_X, breastCancer_training_y, breastCancer_test_X, breastCancer_test_y, "breast Cancer SVM ")
    
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y, title="breast cancer learning curve SVM")


def winequalitySVM():
    winequality = pd.read_csv('winequality.csv', sep=',')

    X = winequality.values[:, 0:-1]
    y = winequality.values[:, -1]

    winequality_training_X,winequality_test_X, winequality_training_y, winequality_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    svm_c(winequality_training_X, winequality_training_y, 'winequality')
    svm_kernel(winequality_training_X, winequality_training_y, 'winequality')
    
    clf = SVC(C=1, kernel='rbf')
    after_tuned_evaluation(clf, winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "wine quality SVM ")
    plot_learning_curve(clf, winequality_training_X, winequality_training_y, title="winequality learning curve SVM")


# if __name__ == "__main__":
#     if not os.path.exists('images'):
#         os.makedirs('images')

#     breastCancerSVM()
#     winequalitySVM()