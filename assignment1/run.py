from DT import breastCancerDT, winequalityDT
from NN import breastCancerNN, winequalityNN
from KNN import breastCancerkNN, winequalitykNN
from SVM import breastCancerSVM, winequalitySVM
from Boosting import breastCancerboost, winequalityboost
import numpy as np
import os
if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    np.random.seed(2636456)
    breastCancerDT()
    winequalityDT()
    breastCancerNN()
    winequalityNN()
    breastCancerkNN()
    winequalitykNN()
    breastCancerSVM()
    winequalitySVM()
    breastCancerboost()
    winequalityboost()
