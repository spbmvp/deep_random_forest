from math import sqrt
from joblib import Parallel, delayed
import numpy as np
import time

if __name__ == '__main__':
    Parallel(n_jobs=-1)(delayed(sqrt)(i**2) for i in range(10))
