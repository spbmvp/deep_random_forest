import logging as log

import numpy as np
from joblib import Parallel, delayed
from matplotlib.mlab import frange

from ForestsModel import ForestsModel
from LoadSet import LoadSet


def initLogging(logLevel):
    log.basicConfig(format=u'[%(asctime)s] %(levelname)-5s %(filename)s[L%(lineno)3d]# %(message)s ',
                    level=logLevel)
    log.addLevelName(9, "TRACE")


def forestFit(file_name, nameSet):
    log.info("#######Start %s set fitting#########", nameSet)
    log.info("  vi     MeanAcc        Med.Acc        Min acc        Max acc   lambda")
    for lamda in [10 ** pow_lamda for pow_lamda in range(-12, 5, 30)]:
        Parallel(n_jobs=-1)(delayed(in_parallel)(file_name, vi, lamda) for vi in frange(0, 1.0, 0.1))
    log.info("#########Finish %s set fitting##########", nameSet)


def in_parallel(file_name, vi, lamda):
    acc_array = []
    for i in range(100):
        X_train, y_train, X_test, y_test = LoadSet(file_name).getSet()
        cascade_forest = ForestsModel(n_trees_cf=int(np.sqrt(X_train.shape[0])), random_magic_num = 241 + 3*i).get_forests()
        try:
            cascade_forest.fit(X_train, y_train, vi=vi, lamda=lamda)
            pred = cascade_forest.predict(X_test)
            k = 0
            for j in range(len(pred)):
                if pred[j] == y_test[j]:
                    k += 1
            log.debug("Step %d: Accuracy = %f", i, float(k / len(pred)))
            acc_array.append(float(k / len(pred)))
        except:
            log.error("что-то пошло не так на этом этапе")
    med = np.median(acc_array)
    mean = np.mean(acc_array)
    log.info(" %s %s %s %s %s %s", vi, mean, med, np.min(acc_array), np.max(acc_array), lamda)


if __name__ == '__main__':
    initLogging(log.INFO)

    forestFit("TrainData/breast.txt", "Breast")
    forestFit("TrainData/ionosphere_data.txt", "Ion")
    forestFit("TrainData/seeds.txt", "Seeds")
    forestFit("TrainData/mammographic.txt", "Mammographic")
    forestFit("TrainData/parkinsons.data.txt", "Parkinsons")
    forestFit("TrainData/usps.txt", "Usps")

    # X_train, y_train = LoadData("TrainData/mnist_train.txt").getSet()
    # X_test, y_test = LoadData("TrainData/mnist_test.txt").getSet()
    # forestFit(X_train, y_train, X_test, y_test, "Mnist")
