import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def fit_ada_boost(features, labels):
    param_dist = {
        'n_estimators': [50, 100, 300, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        # 'loss': ['linear', 'square', 'exponential']
    }

    pre_gs_inst = RandomizedSearchCV(AdaBoostClassifier(),
                                     param_distributions=param_dist,
                                     cv=3,
                                     n_iter=10,
                                     n_jobs=-1)

    model = pre_gs_inst.fit(features, labels)
    return model


def fit_random_forest(features, labels):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    rf = RandomForestClassifier()
    model = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42,
                               n_jobs=-1)
    # RandomForestRegressor(n_estimators=1000, random_state=42, min_impurity_decrease=5, max_depth=5000)

    model = model.fit(features, labels)
    return model


def get_results(y_true, pred):
    return (y_true == pred).value_counts()


def fit_SVM(features, labels):
    svm = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(svm, param_grid, n_jobs=8, verbose=1)
    model = grid_search.fit(features, labels)
    return model
