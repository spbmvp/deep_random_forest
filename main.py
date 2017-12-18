import logging as log

import numpy as np
from joblib import Parallel, delayed
from matplotlib.mlab import frange
from sklearn.metrics import accuracy_score

from ForestsModel import ForestsModel
from LoadSet import LoadSet


def initLogging(logLevel):
    log.basicConfig(format=u'[%(asctime)s] %(levelname)-5s %(filename)s[L%(lineno)3d]# %(message)s ',
                    level=logLevel)
    log.addLevelName(9, "TRACE")


def forestFit(file_name, nameSet):
    log.info("%s", nameSet)
    log.info("vi\tmean\tmedian\tmin\tmax\tlambda")
    for vi in frange(0.0, 1.0, 0.1):
        Parallel(n_jobs=-1)(delayed(in_parallel)(file_name, vi, lamda) for lamda in
                            [10 ** pow_lamda for pow_lamda in range(-4, 4, 1)])
    log.info("")


def in_parallel(file_name, vi, lamda):
    acc_array = []
    for i in range(1):
        X_train, y_train, X_test, y_test = LoadSet(file_name).getSet()
        cascade_forest = ForestsModel(n_trees_cf=int(np.sqrt(X_train.shape[0])),
                                      random_magic_num=241 + 3 * i).get_forests()
        try:
            cascade_forest.fit(X_train, y_train, vi=vi, lamda=lamda)
            pred = cascade_forest.predict(X_test)
            acc = np.zeros(len(pred))
            for j in range(len(pred)):
                acc[j] = accuracy_score(y_test, pred[j])
                log.log(9,"Step %d forest # %d: Accuracy = %f", i, j, acc[j])
            log.debug("Step %d: Accuracy = %f", i, acc.mean())
            acc_array.append(acc.mean())
        except:
            log.error("что-то пошло не так на этом этапе")
    med = np.median(acc_array)
    mean = np.mean(acc_array)
    log.info("%s\t%s\t%s\t%s\t%s\t%s", vi, mean, med, np.min(acc_array), np.max(acc_array), lamda)


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
