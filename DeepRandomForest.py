import copy
import logging as log

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


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
    n_estimator = 0
    classes = 0
    list_weight = []
    max_score = 0
    _eps = 0.0001

    def __init__(self, cf_model, mgs_model, eps=0.0001, **kwargs):
        self._cascade_levels = []
        self._current_level = 0
        self._eps = eps
        # инициализация слоя mgs
        if mgs_model:
            self._mgs_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                                    for estimator in mgs_model]
            self._widows_size = kwargs.setdefault("windows_size", 1)

        # инициализация каскадов
        self._cf_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model]
        self.n_estimator = [estimator.n_estimators for estimator in self._cf_estimators]

    def fit(self, X, y=None, lamda=10 ** -4, vi=1):
        self._len_X = len(X)
        self.classes = np.unique(y)
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        else:
            X = np.array([X_i.ravel() for X_i in X])
        self.cf_fit(X, y, lamda, vi)

    def predict(self, X):
        self._len_X = len(X)
        if self._mgs_estimators is not None:
            X = self.mgs_predict(X)
        else:
            X = np.array([X_i.ravel() for X_i in X])
        return self.cf_predict(X)

    def mgs_fit(self, X, y=None):
        log.debug('Обучение mgs началось X_shape = ' + X.shape)
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        y = np.hstack([y for _ in range(int(len(X) / len(y)))])
        [mgs.fit(X, y) for mgs in self._mgs_estimators]
        new_X = np.hstack([cross_val_predict(
            mgs,
            X,
            y,
            cv=3,
            method='predict_proba',
            n_jobs=-1,
        ) for mgs in self._mgs_estimators])
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        log.debug('Обучение mgs закончено X_shape = %s', new_X.shape)
        return new_X

    def cf_fit(self, X, y=None, lamda=10 ** -4, vi=1):
        log.log(9, 'Обучение cf началось')
        X_old = X_new = X
        while True:
            log.log(9, 'Обучение уровня %d cf началось. X_shape = %s', self._current_level, X_new.shape)
            predict = []
            for estimator in self._cf_estimators:
                estimator.fit(X_new, y)
                predict.append(self.cross_val(estimator, X_new, y))
                pass
            I = np.zeros((self._len_X, len(self.classes)))
            for i in range(self._len_X):
                I[i][y[i]] = 1
            trees_weight = np.zeros((len(self.n_estimator), self.n_estimator[0]))
            for i in range(len(self.n_estimator)):
                trees_weight[i] = (self.calculate_weight_tree(I, lamda, predict[i], vi, i))
                log.log(9, 'Веса деревьев леса № %d получены', i)
                log.debug('Веса деревьев леса № %d: %s', i, trees_weight[i])
                v = self.v_calc(predict[i], trees_weight[i])
                X_old = np.hstack((X_old, v))
            score = np.ones(len(self._cf_estimators))
            for i in range(len(self._cf_estimators)):
                score[i] = (accuracy_score(y, predict[i].mean(axis=0).argmax(axis=1)))
                log.log(9, 'Score IDF леса № %d = %s', i, score[i])
            score = score.mean()
            log.debug("Уровнь %d, Score IDF = %s", self._current_level, score)
            if self.max_score <= score and score - self.max_score >= self._eps:
                self.max_score = score
            else:
                log.log(9,
                        'Обучение уровня %d IDF закончилось X_shape = %s дал худший результат чем предыдущая итерация',
                        self._current_level, X_new.shape)
                break
            log.log(9, 'Обучение уровня %d cf закончилось X_shape = %s', self._current_level, X_old.shape)
            X_new = X_old
            self._cascade_levels.append(copy.deepcopy(self._cf_estimators))
            self.list_weight.append(trees_weight)
            self._current_level += 1
        log.log(9, 'Обучение cf закончилось')

    def calculate_weight_tree(self, I, lamda, predict, vi, i):
        L = np.zeros((self.n_estimator[i], self.n_estimator[i]))
        P = np.zeros((self.n_estimator[i], self.n_estimator[i]))
        b = np.ones(self.n_estimator[i])
        for k in range(self.n_estimator[i]):
            b[k] = sum(sum(predict[:][:][k] * I))
            for j in range(self.n_estimator[i]):
                P[k][j] = sum(sum(np.dot(predict[:][:][k], predict[:][:][j].transpose())))
        np.fill_diagonal(L, lamda)
        a = L + P

        log.log(9, 'Матрица L = \n %s', L)
        log.log(9, 'Матрица P = \n %s', P)
        log.log(9, 'Матрица B = \n %s', b)
        tree_weight = np.linalg.solve(a, b)
        tree_weight = (1.0 - vi) * np.ones(self.n_estimator[i]) / self.n_estimator[i] + vi * tree_weight
        return tree_weight / sum(tree_weight)

    def mgs_predict(self, X):
        log.log(9, 'Тестирование mgs началось X_shape = %s', X.shape)
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        log.log(9, 'Каскады протестированы X_shape = %s', X.shape)
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        log.log(9, 'Тестирование mgs закончено X_shape = %s', new_X.shape)
        return new_X

    def cf_predict(self, X):
        X_new = X_old = X
        for i in range(len(self._cascade_levels)):
            predict = []
            for j in range(len(self._cascade_levels[i])):
                predict_forest = np.zeros((len(self._cascade_levels[i][j].estimators_), len(X), len(self.classes)))
                k = 0
                for forest in self._cascade_levels[i][j].estimators_:
                    predict_forest[k] = forest.predict_proba(X_new)
                    k += 1
                v = self.v_calc(predict_forest, self.list_weight[i][j])
                X_old = np.hstack((X_old, v))
                predict.append(predict_forest)
            X_new = X_old
        res = np.ones((len(self._cf_estimators), len(X)))
        for i in range(len(self._cf_estimators)):
            res[i] = predict[i].mean(axis=0).argmax(axis=1)
        return res

    def windows_sliced(self, X: np.array):
        if self._widows_size * X.shape[1] < 1:
            log.log(9, "Окно слишком маленькое. Автоматически установлено равное 1 признаку")
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

    def v_calc(self, prediction, weight):
        classes_pred = []
        for j in range(self.classes.size):
            classes_pred.append(
                np.dot(weight,
                       prediction[:, :, j]))
        classes_pred = np.array(classes_pred).transpose()
        return classes_pred

    def cross_val(self, estimator, X, y):
        est = copy.deepcopy(estimator)
        predict = []
        step = int(len(X) / 3)
        group_1 = list(range(step))
        group_2 = list(range(step, 2 * step))
        group_3 = list(range(2 * step, len(X)))
        split = [[list(np.hstack((group_2, group_3))), group_1],
                 [list(np.hstack((group_1, group_3))), group_2],
                 [list(np.hstack((group_2, group_1))), group_3]]
        for train, test in split:
            est.fit(X[train], y[train])
            predict_group = []
            for forest in est.estimators_:
                predict_group.append(forest.predict_proba(X[test]))
            predict.append(np.array(predict_group))
        p0 = predict[0].transpose(1, 0, 2)
        p1 = predict[1].transpose(1, 0, 2)
        p2 = predict[2].transpose(1, 0, 2)
        return np.vstack((p0, p1, p2)).transpose(1, 0, 2)
