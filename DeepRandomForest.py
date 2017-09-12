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
    _len_X = 0
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
        self._len_X = len(X)
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        else:
            X = np.array([X_i.ravel() for X_i in X])
        self.cf_fit(X, y)

    def predict(self, X):
        self._len_X = len(X)
        if self._mgs_estimators is not None:
            X = self.mgs_predict(X)
        else:
            X = np.array([X_i.ravel() for X_i in X])
        return self.cf_predict(X)

    def mgs_fit(self, X, y=None):
        print('Обучение mgs началось')
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        y = np.hstack([y for _ in range(int(len(X) / len(y)))])
        [mgs.fit(X, y) for mgs in self._mgs_estimators]
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        print('Обучение mgs закончено X_shape = ', new_X.shape)
        return new_X

    def cf_fit(self, X, y=None):
        print('Обучение cf началось')
        while self._current_level != 2:
            predict = []
            estimators = self._cf_estimators.copy()
            for estimator in estimators:
                estimator.fit(X, y)
                predict.append(estimator.predict_proba(X))
            X = np.hstack([X] + predict)
            self._cascade_levels.append(estimators)
            self._current_level += 1
            print('Обучение уровня ', self._current_level, ' cf закончилось')
        print('Обучение cf закончилось')

    def mgs_predict(self, X):
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        print('Обучение mgs закончено X_shape = ', new_X.shape)
        return new_X

    def cf_predict(self, X):
        for level in self._cascade_levels:
            predict = []
            for estimator in level:
                predict.append(estimator.predict_proba(X))
            X = np.hstack([X] + predict)
            return np.array(predict).mean(axis=0).argmax(axis=1)

    def windows_sliced(self, X: np.array):
        if self._widows_size * X.shape[1] < 1:
            print("Окно слишком маленькое. Автоматически установлено равное 1 признаку")
            windows_size = 1
        else:
            windows_size = int(self._widows_size * X.shape[1])
        if X.ndim == 2:
            new_X = np.vstack([X[:, i:i + windows_size]
                               for i in range(X.shape[1] - windows_size + 1)])
            return new_X
        elif X.ndim == 3:
            new_X = np.vstack([X[:, j: j + windows_size, i: i + windows_size]
                               for i in range(X.shape[1] - windows_size + 1)
                               for j in range(X.shape[2] - windows_size + 1)])
            return new_X.reshape(new_X.shape[0], new_X.shape[1] * new_X.shape[2])
