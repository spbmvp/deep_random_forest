from ForestsModel import ForestsModel
from UspsLoad import Usps
from MnistLoad import Mnist


if __name__ == '__main__':
    X, y = Usps().getSet(1000)
    z, y_z = Mnist().getSet(100)
    cascade_forest = ForestsModel().get_forests()
    k, y = cascade_forest.stream(X, y, z, y_z)
