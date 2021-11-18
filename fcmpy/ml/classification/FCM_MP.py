"""
New Idea: Inverse learning and topology post-optimization in Long-term Cognitive Network
reference:Deterministic learning of hybrid Fuzzy Cognitive Maps and network
reduction approaches
Gonzalo Nápoles a,b,∗, Agnieszka Jastrzębska c, Carlos Mosquera a, Koen Vanhoof a,
Władysław Homenda c

"""
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import os
import csv
import sys
import gc


def read_arff(f, **kwargs):
    '''
    reading from arff file
    :param f: filename
    :param kwargs: dictionary of paramters
    :return: X,Y and labels (classes)
    '''
    data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)  # dataset
    class_att = meta.names()[-1]
    Y = df[class_att]

    # replacing labels
    labels = np.unique(Y)
    mapping = pd.Series([x[0] for x in enumerate(labels)], index=labels)
    Y = Y.map(mapping)  # target

    X = np.array(df.iloc[:, :-1], dtype=np.float64)

    return X, Y, labels


L, U = .1, .9


def logit(X, slope=1.0, h=0.0, q=1.0, v=1.0, **kwargs):
    return 1.0 / np.power(1.0 + q * np.exp(-slope * (X - h)), 1.0 / v)


def expit(Y, slope=1.0, h=0.0, q=1.0, v=1.0, **kwargs):
    return h - np.log((np.power(Y, -v) - 1.0) / q) / slope


class Model(object):
    '''
    FCM model
    '''
    def __init__(self, T=None, p=[], rule=0, b1=1.0, L=0, **kwargs):
        self.T, self.p = T, p
        self.weights = None
        self.p_out = p.copy()

        if rule == 0:
            self.rule = lambda a, W, out: logit(np.dot(a, W), *p)
        elif rule == 1:
            self.rule = lambda a, W, out: b1 * logit(np.dot(a, W), *p) + (1. - b1) * out[0]
        else:
            self.rule = lambda a, W, out: b1 * logit(np.dot(a, W), *p) + (1. - b1) * logit(
                W.diagonal() * np.sum(out[-L - 1:-1], axis=0), *p)

    def train(self, X_train, Y_train, verbose=False, callback=None, **kwargs):
        '''

        :param X_train: features
        :param Y_train: labels
        :param verbose:
        :param callback:
        :param kwargs: dict of params
        writes calculated weight matrix to the self.weights
        '''
        W = self.map(X_train)
        W[np.isnan(W)] = 0
        self.weights = W, np.random.random((X_train.shape[1], Y_train.shape[1]))

        _, out = self.test(X_train)

        # pseudo-inverse learning
        W_out = np.dot(np.linalg.pinv(out[-1]), expit(Y_train * (U - L) + L, *self.p))
        # normalizing the features importance[-1,1]
        mx = np.max(np.abs(W_out))
        if mx > 1:
            W_out /= mx
            self.p_out[0] *= mx
            self.p_out[1] /= mx
        # normalizing weights values importance[-1,1]
        mxW = np.max(np.abs(W))
        if mxW > 1:
            W /= mxW


        self.weights = W, W_out

    def test(self, X, **kwargs):
        '''
        testing obtained weight tmatrix
        :param X:
        :param kwargs: dict of params
        :return:
        '''
        W, W_out = self.weights
        out = [X]

        for t in range(1, self.T + 1):
            out.append(self.rule(out[t - 1], W, out))

        z = logit(np.dot(out[-1], W_out), *self.p)
        return (z - L) / (U - L), out

    def map(self, X_train, **kwargs):
        '''
        returns Weight matrix
        :param X_train: features
        :param kwargs: dict of params
        :return:
        '''
        n, m = X_train.shape
        T1 = np.sum(X_train, axis=0)
        T3 = np.sum(X_train ** 2, axis=0)

        W = np.random.random((m, m))
        for i in range(0, m):
            for j in range(0, m):
                d = n * T3[i] - T1[i] ** 2
                if d != 0:
                    W[i, j] = (n * np.sum(X_train[:, i] * X_train[:, j]) - T1[i] * T1[j]) / d
        return W


def cross_val(f, folds=10, **kwargs):
    '''
    cross valudation (loss)
    :param f: file
    :param folds: number of fold (def 10)
    :param kwargs: dict of parameters
    :return:
    '''
    X, Y, labels = read_arff(f)
    mse = []
    n_outputs = kwargs["M"]  # output variables
    if X.min() < 0. or X.max() > 1.:
        print("Numerical values need to be normalized.")
        raise Exception

    if kwargs["T"] is None:
        kwargs["T"] = X.shape[1] - n_outputs  # n_inputs

    seed = kwargs.get('seed', 1)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=np.random.RandomState(seed))

    train_error = 0
    test_error = 0

    start = datetime.now()
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        model = Model(**kwargs)
        model.train(X_train[:, :-n_outputs], X_train[:, -n_outputs:].reshape(-1, 1), **kwargs)

        X_hat, _ = model.test(X_train[:, :-n_outputs])  # training error
        X_true = X_train[:, -n_outputs:].reshape(-1, 1)

        fold_train_error = np.sum(np.power(X_true - X_hat, 2) / n_outputs) / len(X_true)
        train_error += fold_train_error / folds

        y_hat, _ = model.test(X_test[:, :-n_outputs])
        y_true = X_test[:, -n_outputs:]

        fold_test_error = np.sum(np.power(y_true - y_hat, 2) / n_outputs) / len(y_true)
        test_error += fold_test_error / folds

    elapsed = datetime.now() - start

    slope, h, _, _ = kwargs["p"]
    return {
        # arguments
        "b1": "%.2f" % kwargs["b1"], "L": kwargs["L"], "slope": "%.2f" % slope, "h": "%.2f" % h,

        # metrics
        "train_error": train_error,
        "test_error": test_error,
        "training_time": elapsed.total_seconds() / folds,
        "weights":model.weights[0],
        "importance":model.weights[1]
        
    }


def run(**params):
    
    '''
    runs the whole algorithm for you, the only thing we ask is the parameters. Most of them can be default
    :param params:
    params = {
    'L':0, # For reasoning rule 3", default=0
    'M':1, #Output variables", default=1
    'T':None, # FCM Iterations default=None
    'b1':1.0,  # type=float, default=1.0
    'folds':10, # Number of folds in a (Stratified)K-Fold"
     'output':'./output.csv', # Output csv file", default="./output.csv"
     'p':[1.0, 1.0, 1.0, 1.0],
     'rule':0, # Reasoning rule", choices=[0, 1, 2], default=0
     'sources':['irisnorm.arff'],
     'verbose':False # store_true, verbosity }

    :return:
    '''
    if 'sources' not in params.keys():
            raise KeyError("You have to provide path to .arff files!")
        
    paramsdefault = {'L':0, 'M':1,
                    'T':None, 'b1':1.0, 'folds':10,
                    'output':'./output.csv', 'p':[1.0, 1.0, 1.0, 1.0],
                    'rule':0, 'verbose':False}

    for key in ['L', 'M', 'T', 'b1', 'folds', 'output', 'p', 'rule','verbose']:
        if key not in params.keys():
            params[key] = paramsdefault[key]

    
    data_sources = [f for f in params['sources'] if f.endswith('.arff')]
    results = []
    with open("./output.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        headers = None
        MSE_hist = []
        for f in data_sources:
            print("Processing {}".format(f))
            # here the model is being created and called
            result = cross_val(f, **params) 
            print(result)
            result['filename'] = f
            results.append(result)
            if headers is None:
                headers = list(result.keys())
                writer.writerow(["name"] + headers)

            MSE_hist += [result["test_error"]]
            writer.writerow([os.path.basename(f)[:-5]] + [result[k] for k in headers])
            
            if params['verbose']:
                print("MSE: %.4f" % MSE_hist[-1])

            csv_file.flush()
            sys.stdout.flush()
            gc.collect()

        print("MSE Average of the model across the %d datasets: %.4f" % (len(data_sources), np.average(MSE_hist)))
    return results 

