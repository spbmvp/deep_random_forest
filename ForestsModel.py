# -*- coding: utf-8 -*-

# В этом файле описывается модель первого сверточного слоя и каскадов

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from DeepRandomForest import DeepRandomForest


class ForestsModel(object):
    """Задает модель лесов в каскадах, и их парраметры
    Parameters
    ----------
    _forest_job : integer, optional (default = -1)
         количество процессоров задействованных для обучения и тестирования деревьев, -1 = все

    _mgs_min_samples_split : integer, optional (default = 2)
        минимальное количество примеров для разбиения дерева в mgs слое

    _mgs_n_estimator : integer, optional (default = 30)
        количество деревьев в mgs слое

    _cascade_n_estimator : integer, list, optional (default = 1000)
        количество деревьев в каскаде cf слое

    _cascade_max_features : integer, string, list, optional (default = [1, 1, 'sqrt', 'sqrt'])
        количество использованных для разбиения признаков в cf слое

    Attributes
    ----------
     _model_multi_grained_level : dict
        модель слоя MGS(MultiGradeScanned)
     _model_cascade_levels : dict
        модель слоя CF(CascadeForest)
    """

    _forest_job = -1
    _model_multi_grained_level = True
    _model_cascade_levels = None

    # MGS
    _mgs_list_tree = [ExtraTreesClassifier,
                      RandomForestClassifier]
    _mgs_min_samples_split = 2
    _mgs_n_estimator = 30
    _mgs_windows_size = 1
    # CF
    _cascade_list_tree = [ExtraTreesClassifier,
                          RandomForestClassifier,
                          ExtraTreesClassifier,
                          RandomForestClassifier]
    _cascade_n_estimator = 1000
    _cascade_max_features = [1, 1, 'sqrt', 'sqrt']

    def __init__(self, n_trees_mgs=30, n_trees_cf=1000):
        if n_trees_cf != 1000:
            self._cascade_n_estimator = n_trees_cf
        if n_trees_mgs != 30:
            self._mgs_n_estimator = n_trees_mgs
        if self._model_multi_grained_level:
            self._model_multi_grained_level = self._generate_mgs_model()
        self._model_cascade_levels = self._generate_cf_model()

    # Возвращает объект класса deep_random_forest в соответствии с заданной моделью
    def get_forests(self):
        deep_random_forest = DeepRandomForest(mgs_model=self._model_multi_grained_level,
                                              cf_model=self._model_cascade_levels,
                                              windows_size=self._mgs_windows_size)
        return deep_random_forest

    @staticmethod
    def _get_class_param(param, number):
        if not isinstance(param, (list, tuple, set)):
            return param
        else:
            return param[number]

    def _generate_mgs_model(self):
        model = []
        for i in range(len(self._mgs_list_tree)):
            model.append(dict(estimators_class=self._mgs_list_tree[i],
                              estimators_params=dict(
                                  n_estimators=self._get_class_param(self._mgs_n_estimator, i),
                                  min_samples_split=self._get_class_param(self._mgs_min_samples_split, i),
                                  n_jobs=self._get_class_param(self._forest_job, i)
                              )))
        return model

    def _generate_cf_model(self):
        model = []
        for i in range(len(self._cascade_list_tree)):
            model.append(dict(estimators_class=self._cascade_list_tree[i],
                              estimators_params=dict(
                                  n_estimators=self._get_class_param(self._cascade_n_estimator, i),
                                  max_features=self._get_class_param(self._cascade_max_features, i),
                                  n_jobs=self._get_class_param(self._forest_job, i)
                              )))
        return model
