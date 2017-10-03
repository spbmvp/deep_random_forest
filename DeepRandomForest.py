import numpy as np
import copy


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

    def __init__(self, cf_model, mgs_model, **kwargs):
        # инициализация слоя mgs
        if mgs_model:
            self._mgs_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                                    for estimator in mgs_model]
            self._widows_size = kwargs.setdefault("windows_size", 1)

        # инициализация каскадов
        self._cf_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model]
        self.n_estimator = [estimator.n_estimators for estimator in self._cf_estimators]

    def fit(self, X, y=None):
        self._len_X = len(X)
        self.classes = np.unique(y)
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
        print('Обучение mgs началось X_shape = ', X.shape)
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
        print('Обучение cf началось X_shape = ', X.shape)
        while self._current_level != 1:
            predict = []
            for estimator in self._cf_estimators:
                estimator.fit(X, y)
                for forest in estimator.estimators_:
                    predict.append(forest.predict_proba(X))
            I = np.zeros((self._len_X, len(self.classes)))
            for i in range(self._len_X):
                I[i][y[i]] = 1
            predict = np.array(predict)
            lamda = 10**-10
            vi = 1
            tree_weight = np.ones(sum(self.n_estimator)) / sum(self.n_estimator)
            for i in range(len(self.n_estimator)):
                summ = sum(self.n_estimator[:i])
                step_tree_weight = tree_weight[summ:summ + self.n_estimator[i]]
                tmp_pred = predict[summ:summ + self.n_estimator[i]]
                for step in range(100):
                    sum_pred = step_tree_weight.reshape(len(step_tree_weight), 1, 1) * tmp_pred
                    sum_pred = sum(sum_pred)
                    grad = 2 * step_tree_weight * lamda + self.sum_pred(tmp_pred, sum_pred - I)
                    t0 = np.argmin(grad)
                    g = np.zeros(self.n_estimator[i]) + (1 - vi) / self.n_estimator[i]
                    g[t0] = 1
                    y0 = 2 / (step + 2)
                    step_tree_weight += y0 * (g - step_tree_weight)
                tree_weight[summ:summ + self.n_estimator[i]] = step_tree_weight
                print('Лес обучен')
            predict = self.pred_calc(predict, tree_weight)
            X = np.hstack([X] + predict)
            self._cascade_levels.append(copy.deepcopy(self._cf_estimators))
            self._current_level += 1
            self.list_weight.append(tree_weight)
            print('Обучение уровня ', self._current_level, ' cf закончилось X_shape = ', X.shape)
        print('Обучение cf закончилось')

    def mgs_predict(self, X):
        print('Тестирование mgs началось X_shape = ', X.shape)
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        print('Каскады протестированы X_shape = ', X.shape)
        if self._widows_size != 1:
            new_X = np.hstack([new_X[i:i + self._len_X] for i in range(0, len(new_X), self._len_X)])
        print('Тестирование mgs закончено X_shape = ', new_X.shape)
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

    def sum_pred(self, pred, classes):
        for i in range(classes.shape[0]):
            for j in range(classes.shape[1]):
                pred[:, i, j] *= classes[i, j]
        pred = np.sum(pred, (1, 2))
        return pred

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
