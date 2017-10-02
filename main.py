from ForestsModel import ForestsModel
from UspsLoad import Usps
from MnistLoad import Mnist

if __name__ == '__main__':
    X, y = Mnist().getSet(1100)
    # z, y_z = Usps().getSet(100)
    z, y_z = X[-100:], y[-100:]
    X, y = X[:-100], y[:-100]
    cascade_forest = ForestsModel().get_forests()
    k, y = cascade_forest.stream(X, y, z, y_z)
