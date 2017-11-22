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
        self.list_weight = []
        X_old = X_new = X
        while True:
            log.log(9, 'Обучение уровня %d cf началось. X_shape = %s', self._current_level, X_new.shape)
            predict = []
            for estimator in self._cf_estimators:
                estimator.fit(X_new, y)
                predict.append(self.cross_val(estimator, X_new, y))
                # for forest in estimator.estimators_:
                #     predict.append(forest.predict_proba(X))
                pass
            I = np.zeros((self._len_X, len(self.classes)))
            for i in range(self._len_X):
                I[i][y[i]] = 1
            predict = np.vstack(predict)
            score = accuracy_score(y, predict.mean(axis=0).argmax(axis=1))
            log.debug("Уровнь %d, Score gcF = %f", self._current_level, score)
            tree_weight = np.ones(sum(self.n_estimator)) / sum(self.n_estimator)
            for i in range(len(self.n_estimator)):
                 self.calculate_weight_tree(I, i, lamda, predict, vi, tree_weight)
            log.log(9, 'Веса деревьев получены')

            predict = self.pred_calc(predict, tree_weight)
            score = accuracy_score(y, np.array(predict).mean(axis=0).argmax(axis=1))
            log.debug("Уровнь %d, Score IDF = %f", self._current_level, score)
            if self.max_score <= score and score - self.max_score >= self._eps:
                self.max_score = score
                X_old = X_new
            else:
                log.log(9, 'Обучение уровня %d IDF закончилось X_shape = %s дал худший результат чем предыдущая итерация',
                    self._current_level, X_new.shape)
                break
            log.log(9, 'Обучение уровня %d cf закончилось X_shape = %s', self._current_level, X_old.shape)
            X_new = np.hstack([X_old] + predict)
            self._cascade_levels.append(copy.deepcopy(self._cf_estimators))
            self._current_level += 1
            self.list_weight.append(tree_weight)
            log.log(9, 'Размер tree_weight = %d, shape X_new = %s', len(tree_weight), X_new.shape)
        log.log(9, 'Обучение cf закончилось')
        log.log(9, 'Запишем веса в файл с именем weight %s _ %s, размер листа %d' % (vi, lamda, len(self.list_weight)))
        if log.getLogger().level <= log.DEBUG:
            f = open("weight_tree/weight" + str(vi) + "_" + str(lamda) + ".txt", 'w')
            for weight in self.list_weight:
                f.write(str(list(weight)).replace(",", "\n").replace("]", "]\n"))
            f.close()

    def calculate_weight_tree(self, I, i, lamda, predict, vi, tree_weight):
        summ = sum(self.n_estimator[:i])
        n_estimator_i_ = self.n_estimator[i]
        step_tree_weight = tree_weight[summ:summ + n_estimator_i_]
        tmp_pred = predict[summ:summ + n_estimator_i_]
        # for step in range(self.n_estimator[i] * 2):
        if vi != 0.0:
            for step in range(500):
                g = np.zeros(n_estimator_i_) + (1 - vi) / n_estimator_i_
                sum_pred = step_tree_weight.reshape(len(step_tree_weight), 1, 1) * tmp_pred
                sum_pred = sum(sum_pred)
                grad = 2 * step_tree_weight * lamda + np.sum(
                    tmp_pred * np.array(sum_pred - I).reshape(1, I.shape[0], I.shape[1]), (1, 2))
                t0 = np.argmin(grad)
                g[t0] += vi
                y0 = 2 / (step + 2)
                step_tree_weight += y0 * (g - step_tree_weight)
        tree_weight[summ:summ + n_estimator_i_] = step_tree_weight

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
            prediction = []
            for j in range(len(self._cascade_levels[i])):
                for forest in self._cascade_levels[i][j].estimators_:
                    prediction.append(forest.predict_proba(X))
            predict = self.pred_calc(np.array(prediction), self.list_weight[i])
            X = np.hstack([X] + predict)
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

    def pred_calc(self, prediction, weight):
        tree_pred = []
        for i in range(len(self.n_estimator)):
            summ = sum(self.n_estimator[:i])
            classes_pred = []
            for j in range(self.classes.size):
                classes_pred.append(
                    np.dot(weight[summ:summ + self.n_estimator[i]],
                           prediction[summ:summ + self.n_estimator[i], :, j]))
            classes_pred = np.array(classes_pred).transpose()
            tree_pred.append(classes_pred)
        return tree_pred

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
        p0=predict[0].transpose(1, 0, 2)
        p1=predict[1].transpose(1, 0, 2)
        p2=predict[2].transpose(1, 0, 2)
        return np.vstack((p0, p1, p2)).transpose(1, 0, 2)
