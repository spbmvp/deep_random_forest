import log
import numpy as np
from joblib import Parallel, delayed
from matplotlib.mlab import frange
from sklearn.metrics import accuracy_score

from ForestsModel import ForestsModel
from LoadSet import LoadSet

logger = log.setup_custom_logger('root', log.INFO)


def forestFit(file_name, name_set):
    logger.info("%s", name_set)
    logger.info("vi\tmean\tmedian\tmin\tmax\tlambda")
    Parallel(n_jobs=-1)(delayed(in_parallel)(file_name, vi, lamda)
                        for vi in frange(0.0, 1.0, 0.2)
                        for lamda in [10 ** pow_lamda for pow_lamda in range(-2, 4, 10)]
                        )
    logger.info("")


def in_parallel(file_name, vi, lamda):
    acc_array = []
    for i in range(10):
        X_train, y_train, X_test, y_test = LoadSet(file_name).getSet()
        cascade_forest = ForestsModel(n_trees_cf=5,
                                      max_depth=None,
                                      # n_trees_cf = int(np.sqrt(X_train.shape[0])),
                                      random_magic_num=241 + 3 * i).get_forests()
        try:
            cascade_forest.fit(X_train, y_train, vi=vi, lamda=lamda)
            pred = cascade_forest.predict(X_test)
            acc = accuracy_score(y_test, pred)
            logger.debug("Step %d: Accuracy = %f", i, acc)
            acc_array.append(acc)
        except:
            logger.error("что-то пошло не так(")
    med = np.median(acc_array)
    mean = np.mean(acc_array)
    logger.info("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f", vi, mean, med, np.min(acc_array), np.max(acc_array), lamda)


if __name__ == '__main__':
    forestFit("TrainData/breast.txt", "Breast")
    # forestFit("TrainData/ionosphere_data.txt", "Ion")
    # forestFit("TrainData/seeds.txt", "Seeds")
    # forestFit("TrainData/mammographic.txt", "Mammographic")
    # forestFit("TrainData/parkinsons.data.txt", "Parkinsons")
    # forestFit("TrainData/usps.txt", "Usps")

    # X_train, y_train = LoadData("TrainData/mnist_train.txt").getSet()
    # X_test, y_test = LoadData("TrainData/mnist_test.txt").getSet()
    # forestFit(X_train, y_train, X_test, y_test, "Mnist")
