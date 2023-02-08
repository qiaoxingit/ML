import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import timeit


def plot_learning_curve(clf, X, y, train_sizes=np.linspace(.1, 1.0, 5), title="Insert Title", ylim=(0.5, 1.01), ):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, n_jobs=4, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('images/'+title)
    plt.clf()
    
    


def plot_tuning_curve(train_scores, test_scores, x, name, title="Insert Title", ylim=(0.5, 1.01)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(name)
    plt.ylabel("Score")

    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(x, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,color="r")
    plt.fill_between(x, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(x, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(x, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('images/'+title)
    plt.clf()

def after_tuned_evaluation(clf,trainX, trainy, testX, testy, title):
    
    start_time = timeit.default_timer()
    clf.fit(trainX, trainy)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    predy = clf.predict(testX)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    
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