from sklearn.ensemble import ExtraTreesClassifier

from LoadSet import LoadSet

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = LoadSet("TrainData/breast.txt").getSet()
    clf = ExtraTreesClassifier(max_depth=5, n_estimators=2)
    clf.fit(X_train, y_train)
    proba = clf.estimators_[1].predict_proba(X_train[:5])
    pass