from random import choice, randint, shuffle
from numpy import array, vstack


class Seeds:
    def __init__(self):
        self.classes = []
        self.labels = []
        self.loadFile()

    def loadFile(self):
        file = open('TrainData/seeds.txt', 'r')
        for line in file.readlines():
            self.parseLine(line)

    def parseLine(self, line: str):
        row = []
        if line[0] == '1':
            for i in range(2, len(line) - 1, 7):
                row.append(float(line[i:i + 6]))
            self.labels.append(0)
        elif line[0] == '2':
            for i in range(2, len(line) - 1, 7):
                row.append(float(line[i:i + 6]))
            self.labels.append(1)
        elif line[0] == '3':
            for i in range(2, len(line) - 1, 7):
                row.append(float(line[i:i + 6]))
            self.labels.append(2)
        self.classes.append(row)

    def getSet(self, count=10):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        shuffle(a)
        for i in a[:count]:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)
