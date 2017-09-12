from ForestsModel import ForestsModel
from Seeds import Seeds
from MnistLoad import Mnist

if __name__ == '__main__':
    data = Mnist()
    X, y = data.getSetTemp()
    cascade_forest = ForestsModel().get_forests()
    cascade_forest.fit(X[:40000], y[:40000])
    pred = cascade_forest.predict(X[40000:])
    k = 0
    for i in range(len(pred)):
        if pred[i] == y[i + 40000]:
            k += 1
    print(float(k/len(pred)))
    # print('y = ', y[9000:9010], 'pred = ', pred[:10])
