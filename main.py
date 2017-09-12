from ForestsModel import ForestsModel
from Seeds import Seeds
from MnistLoad import Mnist

if __name__ == '__main__':
    data = Mnist()
    X, y = data.getSetTemp()
    cascade_forest = ForestsModel().get_forests()
    cascade_forest.fit(X[:9000], y[:9000])
    pred = cascade_forest.predict(X[9000:])
    print('y = ', y[9000:9010], 'pred = ', pred[:10])
