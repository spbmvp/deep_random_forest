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
            self._mgs_estimators = [
                estimator["estimators_class"](**estimator["estimators_params"])
                for estimator in mgs_model
            ]
            self._widows_size = kwargs.setdefault("windows_size", 1)

        # инициализация каскадов
        self._cf_estimators = [
            estimator["estimators_class"](**estimator["estimators_params"])
            for estimator in cf_model
        ]
        self._n_estimator = [
            len(estimator.estimators_) for estimator in self._cf_estimators
        ]

    def stream(self, X, y, z, y_z):
        self._classes = np.unique(y)
        self._len_X = len(X)
        y_tar = []
        mean = self.get_mean_classes(X, y)
        # mean = np.where(mean > 0.2, mean, [0])
        # print(mean[9].reshape(16, 16))
        for number in z:
            y_tar.append([np.linalg.norm(number - classes_mean) for classes_mean in X])
        # y_tar = np.argmin(y_tar, axis=1)
        # y = np.hstack([y, y_tar])
        # X = np.vstack([X, z])
        if self._mgs_estimators is not None:
            X = self.mgs_fit(X, y)
        else:
            X = np.array([X_i.ravel() for X_i in X])
            z = np.array([z_i.ravel() for z_i in z])
        return self.cf_stream(X, y, z, y_z, np.array(y_tar))

    def mgs_fit(self, X, y):
        print("Обучение mgs началось X_shape = ", X.shape)
        if self._widows_size != 1:
            X = self.windows_sliced(X)
        y = np.hstack([y for _ in range(int(len(X) / len(y)))])
        [mgs.fit(X, y) for mgs in self._mgs_estimators]
        new_X = np.hstack([mgs.predict_proba(X) for mgs in self._mgs_estimators])
        if self._widows_size != 1:
            new_X = np.hstack(
                [new_X[i : i + self._len_X] for i in range(0, len(new_X), self._len_X)]
            )
        print("Обучение mgs закончено X_shape = ", new_X.shape)
        return new_X

    def cf_stream(self, X, y, z, y_z, y_pred):
        lamda = 10 ** -2
        percent_mix = 5
        len_percent = int(len(z) * percent_mix / 100)
        y_index = np.random.choice(len(z), len_percent, replace=False)
        # y_index = []
        # y_z_temp = copy.deepcopy(y_pred)
        # for _ in range(len_percent):
        #     tmp_y = y_z_temp.min(axis=1).argmin()
        #     y_index.append(tmp_y)
        #     y_z_temp[tmp_y] = np.ones(y_z_temp[tmp_y].shape) + int(np.max(y_z_temp))
        y_pred = y[np.argmin(y_pred, axis=1)]
        print("Поток DADF стартовал X_shape = ", X.shape)
        self.acur(y_pred, y_z)
        while self._current_level != 20:
            predict_X = []
            predict_z = []

            # тренировочная дата
            y_z_train = y_pred[y_index]
            z_train = z[y_index]
            # дата для тестирования
            z_test = np.delete(z, y_index, axis=0)
            # склеивание
            y = np.hstack([y, y_z_train])
            X = np.vstack([X, z_train])

            for estimator in self._cf_estimators:
                estimator.fit(X, y)
                # predict_A = Parallel(n_jobs=-1)(
                #     delayed(self.tree_predict)(forest, X) for forest in estimator.estimators_)
                predict_A = [
                    forest.predict_proba(X) for forest in estimator.estimators_
                ]
                # predict_B = Parallel(n_jobs=-1)(
                #     delayed(self.tree_predict)(forest, z) for forest in estimator.estimators_)
                predict_B = [
                    forest.predict_proba(z_test) for forest in estimator.estimators_
                ]
                C = np.vstack(
                    (
                        np.hstack(
                            (
                                np.ones(len(estimator.estimators_)),
                                np.zeros(
                                    len(estimator.estimators_) + len(self._classes)
                                ),
                            )
                        ),
                        np.hstack(
                            (
                                np.zeros(len(estimator.estimators_)),
                                np.ones(len(estimator.estimators_)),
                                np.zeros(len(self._classes)),
                            )
                        ),
                        np.hstack(
                            (
                                -1
                                * np.mean(predict_A, axis=1).transpose()
                                / (len(estimator.estimators_) * len(X)),
                                np.mean(predict_B, axis=1).transpose() / (len(z)),
                                np.diag(-1 * np.ones(len(self._classes))),
                            )
                        ),
                        np.hstack(
                            (
                                np.diag(np.ones(2 * len(estimator.estimators_))),
                                np.zeros(
                                    (2 * len(estimator.estimators_), len(self._classes))
                                ),
                            )
                        ),
                    )
                )
                G = np.diag(
                    np.hstack(
                        (
                            lamda + np.zeros(2 * len(estimator.estimators_)),
                            np.ones(len(self._classes)),
                        )
                    )
                )
                a = np.zeros(2 * len(estimator.estimators_) + len(self._classes))
                b = np.hstack(
                    (
                        [1, 1],
                        np.zeros(2 * len(estimator.estimators_) + len(self._classes)),
                    )
                )
                quad_prog = solve_qp(
                    G, a, C.transpose(), b, meq=2 + len(self._classes)
                )[0][: -len(self._classes)]
                weight_X = quad_prog[: len(estimator.estimators_)]
                weight_z = quad_prog[len(estimator.estimators_) :]
                predict_X.append(
                    np.sum(predict_A * weight_X.reshape(len(weight_X), 1, 1), axis=0)
                )
                predict_z.append(
                    np.sum(predict_B * weight_z.reshape(len(weight_z), 1, 1), axis=0)
                )

            y_z = np.hstack((np.delete(y_z, y_index), y_z[y_index]))
            y_pred = np.hstack(
                (
                    np.array(predict_z).mean(axis=0).argmax(axis=1),
                    np.array(predict_X).mean(axis=0).argmax(axis=1)[-len_percent:],
                )
            )

            X = np.hstack(
                [X[:-len_percent]] + list(np.array(predict_X)[:, :-len_percent])
            )
            z = np.vstack(
                (
                    np.hstack([z_test] + predict_z),
                    np.hstack([z_train] + list(np.array(predict_X)[:, -len_percent:])),
                )
            )
            y = y[:-len_percent]

            # y_pred = []
            # for number in z:
            #     y_pred.append([np.linalg.norm(number - classes_mean) for classes_mean in X])
            # y_pred = y[np.argmin(y_pred, axis=1)]

            # y_pred = []
            # mean = self.get_mean_classes(X, y)
            # for number in z:
            #     y_pred.append([np.linalg.norm(number - classes_mean) for classes_mean in mean])
            # y_pred = np.argmin(y_pred, axis=1)

            y_tmp = np.vstack(
                (
                    np.array(predict_z).mean(axis=0),
                    np.array(predict_X).mean(axis=0)[-len_percent:],
                )
            )
            y_index = []
            for _ in range(len_percent):
                tmp_y = y_tmp.max(axis=1).argmin()
                y_index.append(tmp_y)
                y_tmp[tmp_y] = np.zeros(y_tmp[tmp_y].shape)

            # y_index = np.random.choice(len(z), len_percent, replace=False)

            self._current_level += 1
            print(
                "Обучение уровня ",
                self._current_level,
                " cf закончилось X_shape = ",
                X.shape,
            )

            # print(np.where(z[30, :256].reshape(16, 16) != 0, ['@'], [' ']))
            # print(y_pred[30])
            self.acur(y_pred, y_z)
        print("Обучение cf закончилось")
        return y_pred, y_z

    def windows_sliced(self, X: np.array):
        if self._widows_size * X.shape[1] < 1:
            print("Окно слишком маленькое. Автоматически установлено равное 1 признаку")
            windows_size = 1
        else:
            windows_size = int(self._widows_size * X.shape[1])
        if X.ndim == 2:
            new_X = np.vstack(
                [
                    X[:, i : i + windows_size]
                    for i in range(X.shape[1] - windows_size + 1)
                ]
            )
            return new_X
        elif X.ndim == 3:
            new_X = np.vstack(
                [
                    X[:, j : j + windows_size, i : i + windows_size]
                    for i in range(X.shape[1] - windows_size + 1)
                    for j in range(X.shape[2] - windows_size + 1)
                ]
            )
            return new_X.reshape(new_X.shape[0], new_X.shape[1] * new_X.shape[2])

    def get_mean_classes(self, X: np.array, y: np.array):
        return np.vstack(
            [np.mean(X[np.argwhere(y == i)], axis=0) for i in self._classes]
        )

    def acur(self, x, y):
        k = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                k += 1
        print("acuracy = ", 100 * (k / len(x)))
