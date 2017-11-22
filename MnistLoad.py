from numpy import array


class Mnist:
    def __init__(self):
        self.test_classes = []
        self.test_labels = []
        self.train_classes = []
        self.train_labels = []

    def loadFile(self):
        fileTrainImage = open('./TrainData/train-images.idx3-ubyte', 'rb')
        fileTrainImage.read(16)
        fileTrainLabel = open('./TrainData/train-labels.idx1-ubyte', 'rb')
        fileTrainLabel.read(8)
        fileTestImage = open('./TrainData/t10k-images.idx3-ubyte', 'rb')
        fileTestImage.read(16)
        fileTestLabel = open('./TrainData/t10k-labels.idx1-ubyte', 'rb')
        fileTestLabel.read(8)
        self.train_classes = self._readFile(fileTrainImage, 28 * 28)
        self.test_classes = self._readFile(fileTestImage, 28 * 28)
        self.train_labels = self._readFile(fileTrainLabel, 1)
        self.test_labels = self._readFile(fileTestLabel, 1)
        return self

    def writeFile(self):
        set_name = 'TrainData/mnist'
        file_name_train = set_name + '_train.txt'
        file_name_test = set_name + '_test.txt'
        self._writeArrToFile(file_name_train, self.train_classes, self.train_labels)
        self._writeArrToFile(file_name_test, self.test_classes, self.test_labels)
        return self

    def _writeArrToFile(self, file_name_train, classes, labels):
        file = open(file_name_train, 'w')
        for i in range(len(classes)):
            for item in labels[i]:
                file.write(str(item) + " ")
            for item in classes[i]:
                file.write(str(item) + " ")
            file.write("\n")
        file.close()

    def _readFile(self, file, size):
        arr = []
        byte = file.read(1)
        while byte != b'':
            temp = []
            for _ in range(size):
                try:
                    temp.append(ord(byte))
                except Exception:
                    print("Что-то пошло не так")
                byte = file.read(1)
                if byte == b'':
                    break
            arr.append(temp)
        return arr

    def getSet(self):
        X_set = []
        y_set = []
        a = list(range(len(self.classes)))
        for i in a:
            X_set.append(self.classes[i])
            y_set.append(self.labels[i])
        return array(X_set), array(y_set)