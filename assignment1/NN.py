import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util import plot_learning_curve, plot_tuning_curve, after_tuned_evaluation, load_winequality, load_breastCancer
from sklearn.model_selection import train_test_split, cross_validate,ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

def hidden_layers(trainX, trainy, title):
    hidden_layer_sizes = np.linspace(1, 30, 8, dtype=int)
    test_scores = []
    train_scores = []
    for i in hidden_layer_sizes:
        clf = MLPClassifier(hidden_layer_sizes=i, learning_rate_init=0.01)
        result = cross_validate(clf, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_scores.append(result['train_score'])
        test_scores.append(result['test_score'])  

    plot_tuning_curve(train_scores, test_scores, hidden_layer_sizes, "hidden layer","hidden layer vs accuracy of "+title)   


def nn_learning_rate(trainX, trainy, title):
    learning_rates = np.linspace(1e-3, 0.1, 10)
    test_scores = []
    train_scores = []
    for i in learning_rates:
        clf = MLPClassifier(learning_rate_init=i)
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
    breastCancer_training_X, breastCancer_training_y, breastCancer_test_X,  breastCancer_test_y = load_breastCancer()

    hidden_layers(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    nn_learning_rate(breastCancer_training_X, breastCancer_training_y, 'breastCancer')
    
    clf = MLPClassifier(hidden_layer_sizes=3, learning_rate_init=0.02)

    after_tuned_evaluation(clf, breastCancer_training_X, breastCancer_training_y, breastCancer_test_X, breastCancer_test_y, "breast cancer NN ")
    
    plot_learning_curve(clf, breastCancer_training_X, breastCancer_training_y, title="breast cancer learning curve neural network")
    run_loss_curve(clf, breastCancer_training_X, breastCancer_training_y, "breast cancer loss curve")


def winequalityNN():
    winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y = load_winequality()

    hidden_layers(winequality_training_X, winequality_training_y,
           'winequality')
    nn_learning_rate(winequality_training_X, winequality_training_y,
           'winequality')
    
    clf = MLPClassifier(hidden_layer_sizes=10, learning_rate_init=0.01)

    after_tuned_evaluation(clf, winequality_training_X, winequality_training_y, winequality_test_X, winequality_test_y, "wine quality NN ")
    
    plot_learning_curve(clf, winequality_training_X, winequality_training_y, title="winequality learning curve neural network")
    run_loss_curve(clf, winequality_training_X, winequality_training_y, "winequality loss curve")

# if __name__ == "__main__":
#     if not os.path.exists('images'):
#         os.makedirs('images')

#     breastCancerNN()
    # winequalityNN()
    