import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm

import torch.nn as nn
import torch
from torch.autograd import Variable
import sys



def fit_ada_boost(features, labels):
    param_dist = {
        'n_estimators': [50, 100, 300, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        #'loss': ['linear', 'square', 'exponential']
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

def fit_SVM(features, labels):
    svm = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(svm, param_grid, n_jobs = 8, verbose=1)
    model = grid_search.fit(features, labels)
    return model

def fit_KNN(features, labels, withTuning):
    if withTuning:
        param_grid = {'n_neighbors': list(range(1, 30)), 'p': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        model = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', n_jobs=8, verbose=1, cv=2)

    else:
        model = KNeighborsClassifier(n_neighbors=14)

    model.fit(features, labels)

    return model

def fit_gradient_boosting(features, labels, withTuning):

    if withTuning:
        param_grid = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                      'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750],
                      'max_depth': [2, 3, 4, 5, 6, 7]}

        model = GridSearchCV(
            estimator=GradientBoostingClassifier(min_samples_split=2,
                                                 min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=10),
            param_grid=param_grid, scoring='accuracy', n_jobs=4, iid=False, cv=5)
    else:
        model = GradientBoostingClassifier(learning_rate=0.001, n_estimators=500,max_depth=3, min_samples_split=2, min_samples_leaf=2, subsample=1,max_features='sqrt', random_state=10)

    model.fit(features, labels)

    return model

def fit_LSTM(features, labels, withTuning):

    INPUT_SIZE = 17
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    param_grid = {'learning_rate': [0.001, 0.01, 0.1],
                  'num_epochs': [25, 50, 100]}

    # param_grid = {'learning_rate': [0.001, 0.05, 0.01],
    #               'num_epochs': [2]}

    learning_rate = 0.001
    num_epochs = 50

    features = features.to_numpy()
    labels = labels.to_numpy()

    # Simple split 80:20

    sizeTrain = int(len(labels) * 20 / 100)
    sizeTest = len(labels) - sizeTrain

    XTrain = features[:sizeTrain]
    XTest = features[sizeTest:]

    yTrain = labels[:sizeTrain]
    yTest = labels[sizeTest:]

    XTrain = np.reshape(XTrain, (XTrain.shape[0], 1, XTrain.shape[1]))
    XTest = np.reshape(XTest, (XTest.shape[0], 1, XTest.shape[1]))

    rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

    optimiser = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    hidden_state = None

    best_nn = None
    best_loss = 9999.9
    best_param = None

    all_train_epoches = {}
    all_test_loss = {}

    for learning_rate in param_grid['learning_rate']:
        for num_epochs in param_grid['num_epochs']:

            print(learning_rate, num_epochs)

            cur_nn = None
            cur_loss = 9999.9
            cur_param = None
            cur_epoches = []

            for epoch in tqdm(range(num_epochs)):

                inputs = Variable(torch.from_numpy(XTrain).float())
                labels = Variable(torch.from_numpy(yTrain).float())

                output, hidden_state = rnn(inputs, hidden_state)

                loss = criterion(output.view(-1), labels)
                optimiser.zero_grad()
                loss.backward(retain_graph=True)
                optimiser.step()
                # print('epoch {}, loss {}'.format(epoch,loss.item()))

                cur_nn = output, hidden_state
                cur_loss = loss.item()
                cur_param = learning_rate, num_epochs
                cur_epoches.append(loss.item())

            all_train_epoches[str(learning_rate) + "|" + str(num_epochs)] = cur_epoches

            trueValue = Variable(torch.from_numpy(yTest).float())
            inputs = Variable(torch.from_numpy(XTest).float())

            predicted, b = rnn(inputs, hidden_state)
            test_loss = criterion(predicted.view(-1), trueValue)

            all_test_loss[str(learning_rate) + "|" + str(num_epochs)] = test_loss.item()

            if best_loss > test_loss.item():
                best_loss = test_loss
                best_cnn = cur_nn
                best_param = cur_param


    print("For LSTM: ")
    print("-" * 30)
    print('Best score: {}'.format(best_loss))
    print('Running at (rate: {}, epoch: {})'.format(best_param[0], best_param[1]))

    print("All Train Epoches: ")
    print(all_train_epoches)
    print("All Test Epoches: ")
    print(all_test_loss)

    return best_cnn, best_param

class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)

        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state

def get_results(y_true,pred):
    return (y_true == pred).value_counts()
