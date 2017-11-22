from collections import Counter
from random import shuffle

from numpy import array


class LoadSet:
    def __init__(self, file_name):
        self.data = []
        self.labels = []
        self.X_train=[]
        self.y_train=[]
        self.X_test=[]
        self.y_test=[]
        self.loadFile(file_name)

    def loadFile(self, file_name):
        file1 = open(file_name, 'r')
        for line in file1.readlines():
            line_array = array(line.split(), dtype=str)
            self.data.append(line_array)
        file1.close()
        shuffle(self.data)
        for i in range(len(self.data)):
            self.labels.append(self.data[i][0])
        l_count = Counter(self.labels)
        indexes = {}
        for clas in l_count.keys():
            a_items = []
            for i in range(len(self.labels)):
                if self.labels[i] == clas:
                    a_items.append(i)
            indexes.setdefault(clas, a_items)
        temp=[]
        for clas in l_count.keys():
            for i in indexes[clas][:l_count.get(clas) * 2 // 3 + 1]:
                arr=[]
                for j in range(0, len(self.data[i])):
                    arr.append(self.data[i][j])
                temp.append(arr)
        shuffle(temp)
        for i in range(0, len(temp)):
            self.y_train.append(int(temp[i][0]))
            arr=[]
            for j in range(1, len(temp[i])):
                arr.append(float(temp[i][j]))
            self.X_train.append(arr)
        temp=[]
        for clas in l_count.keys():
            for i in indexes[clas][l_count.get(clas) * 2 // 3 + 1:]:
                arr=[]
                for j in range(0, len(self.data[i])):
                    arr.append(self.data[i][j])
                temp.append(arr)
        shuffle(temp)
        for i in range(0, len(temp)):
            self.y_test.append(int(temp[i][0]))
            arr=[]
            for j in range(1, len(temp[i])):
                arr.append(float(temp[i][j]))
            self.X_test.append(arr)

    def getSet(self):
        return array(self.X_train), array(self.y_train), array(self.X_test), array(self.y_test)