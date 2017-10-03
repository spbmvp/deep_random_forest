from random import shuffle
from numpy import array, zeros


class Usps:
    def __init__(self):
        self.classes = []
        self.labels = []
        self.loadFile()

    def loadFile(self):
        file = open('TrainData/usps_train.txt', 'r')
        file_label = open('TrainData/usps_label.txt', 'r')
        for line in file.readlines():
            self.parseLine(line)
        for line in file_label.readlines():
            self.labels.append(int(line[0:2])-1)


    def parseLine(self, line: str):
        row = []
        for i in range(16):
            row_line = []
            for j in range(0, 159, 10):
                # i2 = float(line[i * 160 + j:i * 160 + j + 9])
                if float(line[i * 160 + j:i * 160 + j + 9]) < 0:
                    row_line.append(0)
                else:
                    row_line.append(1)
            row.append(row_line)
        self.classes.append(row)

    def getSet(self, count=10):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        shuffle(a)
        for i in a:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)
