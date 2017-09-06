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

    _mgs_estimators = []
    _cf_estimators = []
    _mgs_cascade_parameters = []
    _cf__cascade_parameters = []
    _cascade_levels = []
    _current_level = 0

    def __init__(self, cf_model, mgs_model, **kwargs):
        # инициализация слоя mgs
        if mgs_model:
            self._mgs_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                                    for estimator in mgs_model]
        # инициализация каскадов
        self._cf_estimators = [estimator['estimators_class'](**estimator['estimators_params'])
                               for estimator in cf_model]

        pass

    def fit_drf(self):
        pass

    def predict_drf(self):
        pass
