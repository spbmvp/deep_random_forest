from random import shuffle
from numpy import array, zeros


class Mnist:
    def __init__(self):
        self.classes = []
        self.labels = []
        self.loadFile()

    def loadFile(self):
        fileImage = open('./TrainData/train-images.idx3-ubyte', 'rb')
        fileImage.read(16)
        fileLabel = open('./TrainData/train-labels.idx1-ubyte', 'rb')
        fileLabel.read(8)
        for _ in range(2000):
            image = zeros((28, 28), dtype=float)
            for i in range(28):
                for j in range(28):
                    if ord(fileImage.read(1)) > 100:
                        image[i, j] = 1
            self.parseLine(ord(fileLabel.read(1)), image)

    def parseLine(self, label: int, image: array):
        if label == 0:
            self.labels.append(0)
        elif label == 1:
            self.labels.append(1)
        elif label == 2:
            self.labels.append(2)
        elif label == 3:
            self.labels.append(3)
        elif label == 4:
            self.labels.append(4)
        elif label == 5:
            self.labels.append(5)
        elif label == 6:
            self.labels.append(6)
        elif label == 7:
            self.labels.append(7)
        elif label == 8:
            self.labels.append(8)
        elif label == 9:
            self.labels.append(9)
        self.classes.append(image)

    def getSetTemp(self, count=10):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        shuffle(a)
        for i in a[:count]:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)

    def getSet(self, count=10):
        X_28_28, y_set = self.getSetTemp(count)
        X_16_16 = array(zeros((len(y_set), 16, 16)))
        for k in range(len(X_28_28)):
            for i in range(1, 15, 1):
                for j in range(1, 15, 1):
                    tmp = X_28_28[k, (i - 1) * 2:(i - 1) * 2 + 2, (j - 1) * 2:(j - 1) * 2 + 2]
                    if (tmp[0, 0] == 1 and tmp[0, 1] == 1) or (tmp[0, 0] == 1 and tmp[1, 0] == 1) or (
                                    tmp[0, 1] == 1 and tmp[1, 1] == 1) or (tmp[1, 0] == 1 and tmp[1, 1] == 1):
                        X_16_16[k, i, j] = 1
        return X_16_16, y_set
