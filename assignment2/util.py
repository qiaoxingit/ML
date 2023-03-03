import numpy as np
import matplotlib.pyplot as plt
import time

def plot_figure(x, y, xlabel, ylabel, label, title):

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    y_mean = np.mean(y, axis = 1)
    y_std = np.std(y, axis = 1)

    plt.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.1,color="r")
    plt.plot(x, y_mean, label=label)
    plt.legend(loc="best")
    plt.savefig('images/'+title)



