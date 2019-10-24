from math import sqrt
from joblib import Parallel, delayed
import numpy as np
import time

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [1, 1, 1], [9, 2, 1]])
    print(a.max())
    for _ in range(3):
        tmp = a.max(axis=1).argmax()
        a[tmp] = np.zeros(a[tmp].shape)
