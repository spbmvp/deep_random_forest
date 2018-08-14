import copy
import logging

import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

log = logging.getLogger('root')


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
    X = []
    y = []
    score = 0
    len_feature = 0

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
        self._cf_estimators = []
        self.n_estimator = []
        for i in range(len(cf_model)):
            self._cf_estimators.append([estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model[i]])
            self.n_estimator.append(len(self._cf_estimators[i]))

    def clean(self):
        self._cascade_levels = []
        self._current_level = 0
        self.list_weight = []
        self.X = []
        self.y = []
        self._len_X = 0
        self.score = 0
        self.len_feature = 0

    def fit(self, X, y=None, lamda=10 ** -4, vi=1):
        self._len_X = len(X)
        self.classes = np.unique(y)
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        else:
            X = np.array([X_i.ravel() for X_i in X])
        self.len_feature = len(X[0])
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
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        log.debug('Обучение mgs закончено X_shape = %s', new_X.shape)
        return new_X

    def forest_fit(self, forest):
        return forest.fit(self.X, self.y)

    def forest_predict_proba(self, tree):
        return tree.predict_proba(self.X)

    def cf_fit(self, X, y=None, lamda=10 ** -4, vi=1):
        log.log(9, 'Обучение cf началось X_shape = %s', X.shape)
        self.list_weight = []
        X_old = X_new = X
        while True:
            log.log(9, 'Обучение уровня %d cf началось. X_shape = %s', self._current_level, X_new.shape)
            predict = []
            self.X = X_new
            self.y = y

            for i in range(len(self._cf_estimators)):
                predict.append([])
                for estimator_i in self._cf_estimators[i]:
                    estimator_i.fit(X_new, y)
                    # predict.append(estimator.predict_proba(X))
                    predict[i].append(cross_val_predict(estimator_i, X_new, y, cv=3, method='predict_proba', n_jobs=-1))

            I = np.zeros((self._len_X, len(self.classes)))
            for i in range(self._len_X):
                I[i][y[i]] = 1
            predict = np.array(predict)
            tree_equals_weight = [np.ones(n_estimator_i) / n_estimator_i for n_estimator_i in self.n_estimator]
            tree_weight = [np.ones(n_estimator_i) / n_estimator_i for n_estimator_i in self.n_estimator]
            for i in range(len(self.n_estimator)):
                 self.calculate_weight_tree(I, i, lamda, predict, vi, tree_weight)
            log.log(9, 'Веса деревьев получены')
            pred_cf = self.pred_calc(predict, tree_equals_weight)
            predict = self.pred_calc(predict, tree_weight)
            score = accuracy_score(y, np.array(predict).mean(axis=0).argmax(axis=1))
            log.debug("Уровнь %d, Score IDF = %f", self._current_level, score)
            if self.max_score <= score and score - self.max_score >= self._eps:
                self.max_score = score
                X_old = X_new
            else:
                log.log(9, 'Обучение уровня %d IDF закончилось X_shape = %s. Результат не лучше, чем предыдущая итерация',
                    self._current_level, X_new.shape)
                break
            log.log(9, 'Обучение уровня %d cf закончилось X_shape = %s', self._current_level, X_old.shape)
            X_new = np.hstack([X_old[:, :self.len_feature]] + predict)
            self._cascade_levels.append(copy.deepcopy(self._cf_estimators))
            self._current_level += 1
            self.list_weight.append(tree_weight)
            log.log(9, 'Размер tree_weight = %s, shape X_new = %s', np.array(tree_weight).shape, X_new.shape)
        log.log(9, 'Обучение cf закончилось')

    def calculate_weight_tree(self, I, i, lamda, predict, vi, tree_weight):
        step_tree_weight = tree_weight[i]
        tmp_pred = copy.copy(predict[i])
        if vi != 0.0:
            for step in range(100):
                sum_pred = step_tree_weight.reshape(len(step_tree_weight), 1, 1) * tmp_pred
                sum_pred = np.sum(sum_pred, axis=0)
                grad = 2 * step_tree_weight * lamda + self.sum_pred(tmp_pred, sum_pred - I)
                t0 = np.argmin(grad)
                g = np.zeros(self.n_estimator[i]) + (1 - vi) / self.n_estimator[i]
                g[t0] += vi
                y0 = 0.2 / (step + 2)
                step_tree_weight += y0 * (g - step_tree_weight)
        tree_weight[i] = step_tree_weight

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
        predict = 0
        for i in range(len(self._cascade_levels)):
            log.log(9, 'Тестирование уровня %d cf началось. X_shape = %s', i, X.shape)
            prediction = []
            for j in range(len(self._cascade_levels[i])):
                prediction.append([])
                for estimator_i in self._cascade_levels[i][j]:
                    prediction[j].append(estimator_i.predict_proba(X))
            predict = self.pred_calc(np.array(prediction), self.list_weight[i])
            X = np.hstack([X[:, :self.len_feature]] + predict)
        return np.array(predict).mean(axis=0).argmax(axis=1)

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

    def sum_pred(self, pred, classes):
        for i in range(classes.shape[0]):
            for j in range(classes.shape[1]):
                pred[:, i, j] *= classes[i, j]
        pred = np.sum(pred, (1, 2))
        return pred

    def pred_calc(self, prediction, weight):
        tree_pred = []
        for i in range(len(prediction)):
            classes_pred = []
            for j in range(self.classes.size):
                classes_pred.append(
                    np.dot(weight[i], prediction[i][:, :, j]))
            classes_pred = np.array(classes_pred).transpose()
            tree_pred.append(classes_pred)
        return tree_pred
