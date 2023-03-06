import sys
import time

import matplotlib.pyplot as plt
import mlrose_hiive as mlr
import numpy as np
import pandas as pd
from helper import log
from imblearn.over_sampling import RandomOverSampler
from mlrose_hiive import ExpDecay
from sklearn import preprocessing
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (ShuffleSplit, cross_validate,
                                     train_test_split)


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

    train_sizes = (np.linspace(.25, 1.0, 5) * len(trainy)).astype('int')

    train_results =[]
    test_results = []

    for size in train_sizes:
  
        result = cross_validate(estimator, trainX, trainy, cv = ShuffleSplit(n_splits=50, test_size=0.25, random_state=0), n_jobs=4, return_train_score=True)

        train_results.append(result['train_score'])
        test_results.append(result['test_score'])  

    return train_results, test_results


def evaluation(estimator, trainX, trainy, testX, testy, label):

    exp_decay = ExpDecay(init_temp=100, exp_const=0.2, min_temp=0.001)

    #rhc Training
    #rhc_nn = mlr.NeuralNetwork(hidden_nodes=[50, 30], algorithm=algorithm, learning_rate=0.001, max_attempts=200, max_iters=200, curve=True)
    start_time = time.time()
    estimator.fit(trainX, trainy)
    training_time = time.time() - start_time
    nn_loss = estimator.loss
    #nn_fitness_curve = estimator.fitness_curve

    #rhc Predict
    start_time = time.time()
    predy = estimator.predict(testX)
    predict_time = time.time()- start_time

    f1 = f1_score(testy,predy, average='weighted')
    accuracy = accuracy_score(testy,predy)
    precision = precision_score(testy,predy, average='weighted')
    recall = recall_score(testy,predy, average='weighted')


    sys.stdout = open(label +" test_score.txt", "w")

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(predict_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)) 
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("Loss: "+"{:.2f}".format(nn_loss))
    print("*****************************************************")
    sys.stdout.close()

def run_varied_algorithms():
    log('start to run neural network')
    trainX, trainy, testX, testy = load_data()
    exp_decay = ExpDecay(init_temp=100,
                         exp_const=0.1,
                         min_temp=0.001)

    #rhl                     
    nn_rhl = mlr.NeuralNetwork(hidden_nodes=[5, 3], activation='sigmiod',
                           algorithm='random_hill_climb', max_iters=200,
                           bias=True, is_classifier=True, learning_rate=0.01,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=200, random_state=0, curve=False)

    rhl_train_scores, rhl_test_scores = learning_curve(nn_rhl, trainX=trainX, trainy=trainy)
    rhl_train_scores_mean = np.mean(rhl_train_scores, axis=1)
    rhl_train_scores_std = np.std(rhl_train_scores, axis=1)
    rhl_test_scores_mean = np.mean(rhl_test_scores, axis=1)
    rhl_test_scores_std = np.std(rhl_test_scores, axis=1)

    evaluation(nn_rhl, trainX=trainX, trainy=trainy, testX = testX, testy=testy, label = "rhl")

    #sa
    nn_sa = mlr.NeuralNetwork(hidden_nodes=[50, 30], activation='sigmiod',
                           algorithm='simulated_annealing', schedule=exp_decay, max_iters=200,
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=200, random_state=0, curve=False)

    sa_train_scores, sa_test_scores = learning_curve(nn_sa, trainX=trainX, trainy=trainy)
    sa_train_scores_mean = np.mean(sa_train_scores, axis=1)
    sa_train_scores_std = np.std(sa_train_scores, axis=1)
    sa_test_scores_mean = np.mean(sa_test_scores, axis=1)
    sa_test_scores_std = np.std(sa_test_scores, axis=1)

    evaluation(nn_sa, trainX=trainX, trainy=trainy, testX = testX, testy=testy, label = "sa")

    #ga
    nn_ga = mlr.NeuralNetwork(hidden_nodes=[50, 30], activation='sigmiod',
                           algorithm='genetic_alg', schedule=exp_decay, max_iters=200,
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=200, pop_size = 100, mutation_prob=0.2, random_state=0, curve=False)

    ga_train_scores, ga_test_scores = learning_curve(nn_ga, trainX=trainX, trainy=trainy)
    ga_train_scores_mean = np.mean(ga_train_scores, axis=1)
    ga_train_scores_std = np.std(ga_train_scores, axis=1)
    ga_test_scores_mean = np.mean(ga_test_scores, axis=1)
    ga_test_scores_std = np.std(ga_test_scores, axis=1)

    evaluation(nn_ga, trainX=trainX, trainy=trainy, testX = testX, testy=testy, label = "ga")   

    #gd                    
    nn_gd = mlr.NeuralNetwork(hidden_nodes=[50, 30], activation='sigmiod',
                           algorithm='gradient_decent', max_iters=200,
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=200, random_state=0, curve=False) 

    gd_train_scores, gd_test_scores = learning_curve(nn_gd, trainX=trainX, trainy=trainy)
    gd_train_scores_mean = np.mean(gd_train_scores, axis=1)
    gd_train_scores_std = np.std(gd_train_scores, axis=1)
    gd_test_scores_mean = np.mean(gd_test_scores, axis=1)
    gd_test_scores_std = np.std(gd_test_scores, axis=1)

    evaluation(nn_gd, trainX=trainX, trainy=trainy, testX = testX, testy=testy, label = "gd")   

    #polt
    train_sizes = (np.linspace(.25, 1.0, 5) * len(trainy)).astype('int')
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("Score")
    plt.title("training and validation score with different alorithm")
    plt.grid()

    plt.fill_between(train_sizes, rhl_train_scores_mean - rhl_train_scores_std,
                     rhl_train_scores_mean + rhl_train_scores_std, alpha=0.3)
    plt.fill_between(train_sizes, rhl_test_scores_mean - rhl_test_scores_std,
                     rhl_test_scores_mean + rhl_test_scores_std, alpha=0.3)
    plt.plot(train_sizes, rhl_train_scores_mean, 'o-',
             label="rhl training score")
    plt.plot(train_sizes, rhl_test_scores_mean, 'o-',
             label="rhl cross-validation score")

    plt.fill_between(train_sizes, sa_train_scores_mean - sa_train_scores_std,
                     sa_train_scores_mean + sa_train_scores_std, alpha=0.3)
    plt.fill_between(train_sizes, sa_test_scores_mean - sa_test_scores_std,
                     sa_test_scores_mean + sa_test_scores_std, alpha=0.3)
    plt.plot(train_sizes, sa_train_scores_mean, 'o-',
             label="sa training score")
    plt.plot(train_sizes, sa_test_scores_mean, 'o-',
             label="sa cross-validation score")

    plt.fill_between(train_sizes, ga_train_scores_mean - ga_train_scores_std,
                     ga_train_scores_mean + ga_train_scores_std, alpha=0.3)
    plt.fill_between(train_sizes, ga_test_scores_mean - ga_test_scores_std,
                     ga_test_scores_mean + ga_test_scores_std, alpha=0.3)
    plt.plot(train_sizes, ga_train_scores_mean, 'o-',
             label="ga training score")
    plt.plot(train_sizes, ga_test_scores_mean, 'o-',
             label="ga cross-validation score")

    plt.fill_between(train_sizes, gd_train_scores_mean - gd_train_scores_std,
                     gd_train_scores_mean + gd_train_scores_std, alpha=0.3)
    plt.fill_between(train_sizes, gd_test_scores_mean - gd_test_scores_std,
                     gd_test_scores_mean + gd_test_scores_std, alpha=0.3)
    plt.plot(train_sizes, gd_train_scores_mean, 'o-',
             label="gd training score")
    plt.plot(train_sizes, gd_test_scores_mean, 'o-',
             label="gd cross-validation score")  

    plt.legend(loc="best")
    plt.savefig('images/' +" score of neural network with differnt algorithms")
       
    
if __name__ == "__main__":

    run_varied_algorithms()
