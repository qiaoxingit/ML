import mlrose_hiive as mlr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
from mlrose_hiive import ExpDecay
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing


def load_data():
    np.random.seed(2636456)

    breastCancer = pd.read_csv('breastCancer.csv', sep=',')
    X = breastCancer.values[:, 0:-1]
    y = breastCancer.values[:, -1]

    ros = RandomOverSampler(random_state=0)
    X_res, y_res=ros.fit_resample(X, y)

    breastCancer_training_X, breastCancer_test_X, breastCancer_training_y, breastCancer_test_y = train_test_split(
        X_res, y_res, random_state=42, test_size=0.25)

    scaler = preprocessing.StandardScaler()
    scaler_breastCancer_training_x = scaler.fit_transform(breastCancer_training_X)
    scaler_breastCancer_test_x = scaler.fit_transform(breastCancer_test_X)

    return scaler_breastCancer_training_x, breastCancer_training_y, scaler_breastCancer_test_x, breastCancer_test_y

def learning_curve(estimator, trainX, trainy):

    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])

    train_results =[]
    test_results = []

    for iteration in iterations:
  
        result = cross_validate(estimator, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_results.append(result['train_score'])
        test_results.append(result['test_score'])  

    return test_results, test_results

def plot_tuning_curve(iterations, train_scores, test_scores, ylim=(0.5, 1.01)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("iteration")
    plt.ylabel("Score")
    plt.grid()
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(iterations, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,color="r")
    plt.fill_between(iterations, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(iterations, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(iterations, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

def evaluation(trainX, trainy, testX, testy):

    exp_decay = ExpDecay(init_temp=100, exp_const=0.2, min_temp=0.001)

    #rhc Training
    rhc_nn = mlr.NeuralNetwork(hidden_nodes=[50, 30], algorithm="random_hill_climb", learning_rate=0.001, max_attempts=200, max_iters=200, curve=True)
    start_time = time.time()
    rhc_nn.fit(trainX, trainy)
    rhc_training_time = time.time() - start_time
    nn_loss = rhc_nn.loss
    nn_fitness_curve = rhc_nn.fitness_curve

    #rhc Predict
    start_time = time.time()
    predy = rhc_nn.predict(testX)
    rhc_predict_time = time.time()- start_time

    f1 = f1_score(testy,predy, average='weighted')
    accuracy = accuracy_score(testy,predy)
    precision = precision_score(testy,predy, average='weighted')
    recall = recall_score(testy,predy, average='weighted')


    sys.stdout = open(title+ "after_tuned_test_score.txt", "w")

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)) 
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    sys.stdout.close()
