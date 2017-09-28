from ForestsModel import ForestsModel
from Seeds import Seeds
from MnistLoad import Mnist

if __name__ == '__main__':
    data = Mnist()
    X, y = data.getSetTemp()
    cascade_forest = ForestsModel().get_forests()
    k, y = cascade_forest.stream(X[:1000], y[:1000], X[1000:1500], y[1000:1500])
    print(y[450:], k[450:])
