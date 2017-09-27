import numpy as np
import copy
from joblib import Parallel, delayed
from quadprog import solve_qp


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
    _classes = 0
    _n_estimator = 0

    def __init__(self, cf_model, mgs_model, **kwargs):
        # инициализация слоя mgs
        if mgs_model:
            self._mgs_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                                    for estimator in mgs_model]
            self._widows_size = kwargs.setdefault("windows_size", 1)

        # инициализация каскадов
        self._cf_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model]
        self._n_estimator = [len(estimator.estimators_) for estimator in self._cf_estimators]

    def stream(self, X, y, z):
        self._classes = np.unique(y)
        self._len_X = len(X)
        y_tar = []
        mean = self.get_mean_classes(X, y)
        for number in z:
            y_tar.append([np.linalg.norm(number - classes_mean) for classes_mean in mean])
        # y_tar = np.argmin(y_tar, axis=1)
        # y = np.hstack([y, y_tar])
        # X = np.vstack([X, z])
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        else:
            X = np.array([X_i.ravel() for X_i in X])
            z = np.array([z_i.ravel() for z_i in z])
        return self.cf_stream(X, y, z, np.array(y_tar))

    def mgs_fit(self, X, y):
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

    def cf_stream(self, X, y, z, y_z):
        lamda = 0.00000000001
        percent_mix = 10
        len_percent = int(len(z) * percent_mix / 100)
        untagget_y = None
        y_index = []
        for _ in range(len_percent):
            tmp_y = y_z.min(axis=1).argmin()
            y_index.append(tmp_y)
            y_z[tmp_y] = np.ones(y_z[tmp_y].shape) + int(y_z.max())
        y_z_train = y_z.argmin(axis=1)[y_index]
        z_train = z[y_index]
        z_test = np.delete(z, y_index, axis=0)
        y = np.hstack([y, y_z_train])
        X = np.vstack([X, z_train])

        print('Поток DADF стартовал X_shape = ', X.shape)
        while self._current_level != 4:
            predict_X = []
            predict_z = []
            for estimator in self._cf_estimators:
                estimator.fit(X, y)
                # predict_A = Parallel(n_jobs=-1)(
                #     delayed(self.tree_predict)(forest, X) for forest in estimator.estimators_)
                predict_A = [forest.predict_proba(X) for forest in estimator.estimators_]
                # predict_B = Parallel(n_jobs=-1)(
                #     delayed(self.tree_predict)(forest, z) for forest in estimator.estimators_)
                predict_B = [forest.predict_proba(z_test) for forest in estimator.estimators_]
                C = np.vstack((
                    np.hstack((np.ones(len(estimator.estimators_)),
                               np.zeros(len(estimator.estimators_) + len(self._classes)))),
                    np.hstack((np.zeros(len(estimator.estimators_)),
                               np.ones(len(estimator.estimators_)),
                               np.zeros(len(self._classes)))),
                    np.hstack((-1 * np.mean(predict_A, axis=1).transpose() / (len(estimator.estimators_) * len(X)),
                               np.mean(predict_B, axis=1).transpose() / (len(z)),
                               np.diag(-1 * np.ones(len(self._classes))))),
                    np.hstack((np.diag(np.ones(2 * len(estimator.estimators_))),
                               np.zeros((2 * len(estimator.estimators_), len(self._classes)))))
                ))
                G = np.diag(np.hstack((lamda + np.zeros(2 * len(estimator.estimators_)),
                                       np.ones(len(self._classes)))))
                a = np.zeros(2 * len(estimator.estimators_) + len(self._classes))
                b = np.hstack(([1, 1], np.zeros(2 * len(estimator.estimators_) + len(self._classes))))
                quad_prog = solve_qp(G, a, C.transpose(), b, meq=2 + len(self._classes))[0][:-len(self._classes)]
                weight_X = quad_prog[:len(estimator.estimators_)]
                weight_z = quad_prog[len(estimator.estimators_):]
                predict_X.append(np.sum(predict_A * weight_X.reshape(len(weight_X), 1, 1), axis=0))
                predict_z.append(np.sum(predict_B * weight_z.reshape(len(weight_z), 1, 1), axis=0))
            y_tar = np.array(predict_z).mean(axis=0)
            y_index = []
            for _ in range(len_percent):
                tmp_y = y_tar.max(axis=1).argmax()
                y_index.append(tmp_y)
                y_tar[tmp_y] = np.zeros(y_tar[tmp_y].shape)
            y_tar = y_tar.argmax(axis=1)[y_index]
            z_tar = z_test[y_index]
            z_test = np.vstack((np.delete(z_test, y_index, axis=0), z_train))
            X = np.vstack((X[:-len_percent], z_tar))
            y = np.hstack((y[:-len_percent], y_tar))
            X = np.hstack([X] + list(np.hstack((predict_X[:, -len_percent], predict_z[:, y_index]))))
            z = np.hstack([z] + list(np.vstack((np.delete(predict_z, y_index, axis=1), predict_X[:, -len_percent:]))))

            untagget_y = np.array(predict_z).mean(axis=0).argmax(axis=1)
            y[-len(untagget_y):] = untagget_y
            self._current_level += 1
            print('Обучение уровня ', self._current_level, ' cf закончилось X_shape = ', X.shape)
        print('Обучение cf закончилось')
        return untagget_y

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

    def get_mean_classes(self, X: np.array, y: np.array):
        return np.vstack([np.mean(X[np.argwhere(y == i)], axis=0) for i in self._classes])

    @staticmethod
    def tree_predict(tree, X):
        return tree.predict_proba(X)
