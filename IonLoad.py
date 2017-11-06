from random import choice, randint, shuffle
from numpy import array, vstack


class Ion:
    def __init__(self):
        self.classes = []
        self.labels = []
        self.loadFile()

    def loadFile(self):
        file = open('TrainData/ionosphere_data.txt', 'r')
        for line in file.readlines():
            line_array = array(line.split(), dtype=float)
            self.classes.append(line_array[1:])
            self.labels.append(int(line_array[0]))

    def getSet(self, count=351):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        shuffle(a)
        for i in a[:count]:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)
