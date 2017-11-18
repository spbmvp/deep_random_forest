from collections import Counter
from random import shuffle

from numpy import array
set_name = '/fastdir/Study/deep_random_forest/TrainData/parkinsons.data'
file_name = set_name + '.txt'
file_name_train = set_name + '_train.txt'
file_name_test = set_name + '_test.txt'

data = []
labels = []
file1 = open(file_name, 'r')

for line in file1.readlines():
    line_array = array(line.split(), dtype=str)
    data.append(line_array)
    labels.append(int(line_array[0]))
file1.close()
shuffle(data)
l_count = Counter(labels)
indexes = {}
for clas in l_count.keys():
    a_items = []
    for i in range(len(labels)):
        if labels[i] == clas:
            a_items.append(i)
    indexes.setdefault(clas, a_items)

file = open(file_name_train, 'w')
for clas in l_count.keys():
    for i in indexes[clas][:l_count.get(clas)*2//3+1]:
        for j in range(0, len(data[i])):
            file.write(data[i][j] + " ")
        file.write("\n")
file.close()

file = open(file_name_test, 'w')
for clas in l_count.keys():
    for i in indexes[clas][l_count.get(clas)*2//3+1:]:
        for j in range(0, len(data[i])):
            file.write(data[i][j] + " ")
        file.write("\n")
file.close()