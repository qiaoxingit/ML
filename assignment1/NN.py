import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util import plot_learning_curve, plot_tuning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_validate,ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error



def hidden_layers(trainX, trainy, title):
    hidden_layer_sizes = range(1, 10, 1)
    test_scores = []
    train_scores = []
    for i in hidden_layer_sizes:
        clf = MLPClassifier(hidden_layer_sizes=(i,), random_state=0)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, hidden_layer_sizes, "hidden layer","hidden layer vs accuracy of "+title)   


def nn_learning_rate(trainX, trainy, title):
    learning_rates = np.arange(0.001, 0.01, 0.001)
    test_scores = []
    train_scores = []
    for i in learning_rates:
        clf = MLPClassifier(learning_rate_init=i, random_state=0)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, learning_rates, "learning rate","neural network learning rate vs accuracy of "+title)   


def run_loss_curve(clf, trainX, trainy, title):    
    clf.fit(trainX, trainy)

    plt.plot(clf.loss_curve_, label = 'loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Loss curve for neural network "+title)
    plt.legend(loc='best')
    plt.savefig('images/'+title+" Loss curve for neural network")
    plt.clf()


def breastCancerNN():
    breastCancer = pd.read_csv('breastCancer.csv', sep=',')

    X = breastCancer.values[:, 0:-1]
    y = breastCancer.values[:, -1]

    breastCancer_training_X, breastCancer_test_X, breastCancer_training_y, breastCancer_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    hidden_layers(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    nn_learning_rate(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    clf = MLPClassifier(hidden_layer_sizes=(3,), learning_rate_init=0.001, random_state=0)
    
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y, title="breast cancer learning curve neural network")
    run_loss_curve(clf, breastCancer_training_X, breastCancer_training_y, "breast cancer loss curve")


def winequalityNN():
    winequality = pd.read_csv('winequality.csv', sep=',')

    X = winequality.values[:, 0:-1]
    y = winequality.values[:, -1]

    winequality_training_X, winequality_test_X, winequality_training_y, winequality_test_y = train_test_split(
        X, y, random_state=0, test_size=0.25)

    hidden_layers(winequality_training_X, winequality_training_y,
           'breastCancer')
    nn_learning_rate(winequality_training_X, winequality_training_y,
           'breastCancer')
    
    clf = MLPClassifier(hidden_layer_sizes=(5,), learning_rate_init=0.001, random_state=0)
    
    plot_learning_curve(clf, winequality_training_X, winequality_training_y, title="breast cancer learning curve neural network")
    run_loss_curve(clf, winequality_training_X, winequality_training_y, "breast cancer loss curve")

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    breastCancerNN()
    winequalityNN()
    