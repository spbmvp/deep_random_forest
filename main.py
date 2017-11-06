import logging as log

import numpy as np
from matplotlib.mlab import frange

from ForestsModel import ForestsModel
from Seeds import Seeds
from UspsLoad import Usps


def initLogging(logLevel):
    log.basicConfig(format=u'[%(asctime)s] %(levelname)-5s %(filename)s[L%(lineno)3d]# %(message)s ',
                    level=logLevel)


def forestFit(X_train, y_train, X_test, y_test):
    mean = []
    log.info("Start set fitting")
    for vi in frange(0.01, 1, 0.01):
        log.info("vi = %f", vi)
        for i in range(100):
            cascade_forest = ForestsModel().get_forests()
            cascade_forest.fit(X_train, y_train, vi = vi)
            pred = cascade_forest.predict(X_test)
            k = 0
            for j in range(len(pred)):
                if pred[j] == y_test[j]:
                    k += 1
            log.debug("Step %d: Accuracy = %f", i, float(k / len(pred)))
            mean.append(float(k / len(pred)))
        med = np.median(mean)
        mean = np.mean(mean)
        log.info("\n\tMean accuracy = %f\n\tMedian accuracy = %f", mean, med)


if __name__ == '__main__':
    initLogging(log.DEBUG)

    data = Seeds()
    n_set = 210
    X, y = data.getSet(n_set)
    X_train = X[:int(n_set * (2 / 3))]
    y_train = y[:int(n_set * (2 / 3))]
    X_test = X[int(n_set * (2 / 3)):]
    y_test = y[int(n_set * (2 / 3)):]
    forestFit(X_train, y_train, X_test, y_test)

    data = Usps()
    n_set = 2000
    X, y = data.getSet(n_set)
    X_train = X[:int(n_set * (2 / 3))]
    y_train = y[:int(n_set * (2 / 3))]
    X_test = X[int(n_set * (2 / 3)):]
    y_test = y[int(n_set * (2 / 3)):]
    forestFit(X_train, y_train, X_test, y_test)
