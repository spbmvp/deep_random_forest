import numpy as np


class DeepRandomForest(object):
    """Задает модель лесов в каскадах, и их парраметры
    Attributes
    ----------
    _mgs_estimators : list
         список лесов в слое mgs

    _cf_estimators : list
        список лесов в слое cf

    _mgs_cascade_parameters : list
        список параметров для слоя mgs

    _cf__cascade_parameters : list
        список параметров для слоя cf

    _cascade_levels : list
        список всех каскадов

    _current_level : integer
        текущий номер каскада
    """

    _mgs_estimators = None
    _cf_estimators = None
    _cascade_levels = []
    _current_level = 0
    _widows_size = 1

    def __init__(self, cf_model, mgs_model, **kwargs):
        # инициализация слоя mgs
        if mgs_model:
            self._mgs_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                                    for estimator in mgs_model]
            self._widows_size = kwargs.setdefault("windows_size", 1)

        # инициализация каскадов
        self._cf_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model]

    def fit(self, X, y=None):
        X = self.normalize_x(X)
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        self.cf_fit(X, y)

    def predict(self, X):
        X = self.normalize_x(X)
        if self._mgs_estimators is not None:
            X = self.mgs_predict(X)
        self.cf_predict(X)
        predict = 1
        return predict

    def mgs_fit(self, X, y=None):
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        new_X = [mgs.cross_val_predict(X, y) for mgs in self._mgs_estimators]
        return new_X

    def cf_fit(self, X, y=None):
        while True:
            pass


    def mgs_predict(self, X):
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        new_X = [mgs.cross_val_predict(X, y) for mgs in self._mgs_estimators]
        return new_X

    def cf_predict(self, X):
        pass

    def windows_sliced(self, X):

        return X

    @staticmethod
    def normalize_x(X: np.array):
        if X.ndim == 1:
            raise AttributeError('Размерность х должна быть 2')
        elif X.dim == 2:
            return X
        elif X.ndim == 3:
            norm_X = []
            for matrix in X:
                norm_X.append(matrix.ravel())
            return np.array(norm_X)
        else:
            raise AttributeError('Размерность х должна быть 2')
