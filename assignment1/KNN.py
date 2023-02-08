import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
from util import plot_learning_curve, plot_tuning_curve,after_tuned_evaluation
from sklearn.model_selection import train_test_split, cross_validate,ShuffleSplit

def knn_neighbours(trainX, trainy, title):
    neighbours_list = np.linspace(1, 80, 20).astype(int)
    test_scores = []
    train_scores = []
    for i in neighbours_list:
        clf = kNN(n_neighbors=i)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, neighbours_list, "number of neighbours","number of neighbours vs accuracy of " + title)   


def knn_algorithm(trainX, trainy, title):
    algorithm_list = np.array(["auto", "ball_tree", "kd_tree", "brute"])
    test_scores = []
    train_scores = []
    for algo in algorithm_list:
        clf = kNN(algorithm=algo)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    # plot_tuning_curve(train_scores, test_scores, algorithm_list, "algorithm","algorithm vs accuracy of " + title)   
    sys.stdout = open(title+ "algorithm.txt", "w")
    print(algorithm_list)
    print("average training score for each algorithm")
    print(np.mean(train_scores, axis = 1))
    print("============================================")
    print("average vakidation score for each algorithm")
    print(np.mean(test_scores, axis = 1) )
    
    sys.stdout.close()


def breastCancerkNN():
    breastCancer = pd.read_csv('breastCancer.csv', sep=',')

    X = breastCancer.values[:, 0:-1]
    y = breastCancer.values[:, -1]

    breastCancer_training_X, breastCancer_test_X, breastCancer_training_y, breastCancer_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    knn_neighbours(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    knn_algorithm(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    clf = kNN(n_neighbors=30, algorithm="ball_tree")
    after_tuned_evaluation(clf, breastCancer_training_X, breastCancer_training_y, breastCancer_test_X, breastCancer_test_y, "breast Cancer kNN ")
    
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y, title="breast cancer learning curve kNN")


def winequalitykNN():
    winequality = pd.read_csv('winequality.csv', sep=',')

    X = winequality.values[:, 0:-1]
    y = winequality.values[:, -1]

    winequality_training_X,winequality_test_X, winequality_training_y, winequality_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    knn_neighbours(winequality_training_X, winequality_training_y, 'winequality')
    knn_algorithm(winequality_training_X, winequality_training_y, 'winequality')
    
    clf = kNN(n_neighbors=30, algorithm="ball_tree")
    after_tuned_evaluation(clf, winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "wine quality kNN ")
    plot_learning_curve(clf, winequality_training_X, winequality_training_y, title="winequality learning curve kNN")


# if __name__ == "__main__":
#     if not os.path.exists('images'):
#         os.makedirs('images')

#     breastCancerkNN()
#     winequalitykNN()