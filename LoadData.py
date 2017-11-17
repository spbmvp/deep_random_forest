from numpy import array


class LoadData:
    def __init__(self, file_name):
        self.classes = []
        self.labels = []
        self.loadFile(file_name)

    def loadFile(self, file_name):
        file = open(file_name, 'r')
        for line in file.readlines():
            line_array = array(line.split(), dtype=float)
            self.classes.append(line_array[1:])
            self.labels.append(int(line_array[0]))

    def getSet(self):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        for i in a:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)
