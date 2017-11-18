import logging as log
from copy import deepcopy as dc

import numpy as np
from joblib import Parallel, delayed
from matplotlib.mlab import frange

from ForestsModel import ForestsModel
from LoadData import LoadData


def initLogging(logLevel):
    log.basicConfig(format=u'[%(asctime)s] %(levelname)-5s %(filename)s[L%(lineno)3d]# %(message)s ',
                    level=logLevel)


def forestFit(X_train, y_train, X_test, y_test, nameSet):
    log.info("#######Start %s set fitting#########", nameSet)
    log.info("  vi  MeanAcc  Med.Acc  Min acc  Max acc lambda")
    for lamda in [10 ** pow_lamda for pow_lamda in range(-12, 4, 2)]:
        Parallel(n_jobs=10)(delayed(in_parallel)(X_test, X_train, vi, lamda, y_test, y_train) for vi in frange(0, 1, 0.1))
    log.info("#########Finish %s set fitting##########", nameSet)


def in_parallel(X_test, X_train, vi, lamda, y_test, y_train):
    acc_array = []
    for i in range(5):
        cascade_forest = dc(ForestsModel(n_trees_cf=int(X_train.shape[0]), random_magic_num = 241*i)).get_forests()
        cascade_forest.fit(X_train, y_train, vi=vi, lamda=lamda)
        pred = cascade_forest.predict(X_test)
        k = 0
        for j in range(len(pred)):
            if pred[j] == y_test[j]:
                k += 1
        log.debug("Step %d: Accuracy = %f", i, float(k / len(pred)))
        acc_array.append(float(k / len(pred)))
    med = np.median(acc_array)
    mean = np.mean(acc_array)
    log.info(" %s %f %f %f %f %s", vi, mean, med, np.min(acc_array), np.max(acc_array), lamda)


if __name__ == '__main__':
    initLogging(log.INFO)

    X_train, y_train = LoadData("TrainData/breast_train.txt").getSet()
    X_test, y_test = LoadData("TrainData/breast_test.txt").getSet()
    forestFit(X_train, y_train, X_test, y_test, "Breast")

    # X_train, y_train = LoadData("TrainData/ionosphere_data_train.txt").getSet()
    # X_test, y_test = LoadData("TrainData/ionosphere_data_test.txt").getSet()
    # forestFit(X_train, y_train, X_test, y_test, "Ion")

    X_train, y_train = LoadData("TrainData/seeds_train.txt").getSet()
    X_test, y_test = LoadData("TrainData/seeds_test.txt").getSet()
    forestFit(X_train, y_train, X_test, y_test, "Seeds")

    X_train, y_train = LoadData("TrainData/mammographic_train.txt").getSet()
    X_test, y_test = LoadData("TrainData/mammographic_test.txt").getSet()
    forestFit(X_train, y_train, X_test, y_test, "Mammographic")

    X_train, y_train = LoadData("TrainData/parkinsons.data_train.txt").getSet()
    X_test, y_test = LoadData("TrainData/parkinsons.data_test.txt").getSet()
    forestFit(X_train, y_train, X_test, y_test, "Parkinsons")

    X_train, y_train = LoadData("TrainData/usps_train.txt").getSet()
    X_test, y_test = LoadData("TrainData/usps_test.txt").getSet()
    forestFit(X_train, y_train, X_test, y_test, "Usps")
