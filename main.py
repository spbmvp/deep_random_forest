import numpy as np
from ForestsModel import ForestsModel
from Seeds import Seeds
from MnistLoad import Mnist
from UspsLoad import Usps
from Mammographic import Mamm
from copy import deepcopy

if __name__ == '__main__':
    data = Seeds()
    n_set = 210
    mean = []
    for _ in range(50):
        X, y = data.getSet(n_set)
        cascade_forest = ForestsModel().get_forests()
        cascade_forest.fit(X[:int(n_set * (2 / 3))], y[:int(n_set * (2 / 3))])
        pred = cascade_forest.predict(X[int(n_set*(2/3)):])
        k = 0
        for i in range(len(pred)):
            if pred[i] == y[i + int(n_set*(2/3))]:
                k += 1
        print(float(k/len(pred)))
        mean.append(float(k/len(pred)))
    med = np.median(mean)
    mean = np.mean(mean)
    print(mean, med)
    # print('y = ', y[9000:9010], 'pred = ', pred[:10])
