"""
New Idea: Inverse learning and topology post-optimization in Long-term Cognitive Network
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

    def __init__(self, T=None, p=[], rule=0, b1=1.0, L=0, **kwargs):
        self.T, self.p = T, p
        self.weights = None

        if rule == 0:
            self.rule = lambda a, W, out: logit(np.dot(a, W), *p)
        elif rule == 1:
            self.rule = lambda a, W, out: b1 * logit(np.dot(a, W), *p) + (1. - b1) * out[0]
        else:
            self.rule = lambda a, W, out: b1 * logit(np.dot(a, W), *p) + (1. - b1) * logit(
                W.diagonal() * np.sum(out[-L - 1:-1], axis=0), *p)

    def train(self, X_train, Y_train, verbose=False, callback=None, **kwargs):
        # W = np.array(pd.DataFrame(X_train).corr())
        W = self.map(X_train)
        W[np.isnan(W)] = 0
        self.weights = W, np.random.random((X_train.shape[1], Y_train.shape[1]))

        _, out = self.test(X_train)

        # pseudo-inverse learning
        W_out = np.dot(np.linalg.pinv(out[-1]), expit(Y_train * (U - L) + L, *self.p))

        # genetic-algorithm learning
        # W_out = ...

        self.weights = W, W_out

    def test(self, X, **kwargs):
        W, W_out = self.weights
        out = [X]

        for t in range(1, self.T + 1):
            out.append(self.rule(out[t - 1], W, out))

        z = logit(np.dot(out[-1], W_out), *self.p)
        return (z - L) / (U - L), out

    def map(self, X_train, **kwargs):
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
        "weights":model.weights
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--sources", nargs='+', required=True, help="Dataset directory or arff files")
    parser.add_argument("-o", "--output", help="Output csv file", default="./output.csv")
    parser.add_argument("-f", "--folds", type=int, help="Number of folds in a (Stratified)K-Fold", default=10)
    parser.add_argument("-r", "--rule", type=int, help="Reasoning rule", choices=[0, 1, 2], default=0)
    parser.add_argument("-T", type=int, help="FCM Iterations", default=None)
    parser.add_argument("-L", type=int, help="For reasoning rule 3", default=0)
    parser.add_argument("-M", type=int, help="Output variables", default=1)
    parser.add_argument("-b1", type=float, default=1.0)

    parser.add_argument("-p", nargs=4, type=float, help="Params for expit and logit functions",
                        metavar=('slope', 'h', 'q', 'v'), default=[1.0, 1.0, 1.0, 1.0])

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")

    args = parser.parse_args()

    data_sources = [f for f in args.sources if f.endswith('.arff')]

    dirs = [dir for dir in args.sources if os.path.isdir(dir)]
    data_sources += [os.path.join(dir, f) for dir in dirs for f in os.listdir(dir) if f.endswith('.arff')]
    data_sources.sort()

    with open(args.output, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        headers = None
        MSE_hist = []
        for f in data_sources:
            print("Processing {}".format(f))
            # here the model is being created and called
            result = cross_val(f, **vars(args))
            print(result)
            if headers is None:
                headers = list(result.keys())
                writer.writerow(["name"] + headers)

            MSE_hist += [result["test_error"]]
            writer.writerow([os.path.basename(f)[:-5]] + [result[k] for k in headers])

            if args.verbose:
                print("MSE: %.4f" % MSE_hist[-1])

            csv_file.flush()
            sys.stdout.flush()
            gc.collect()

        print("MSE Average of the model across the %d datasets: %.4f" % (len(data_sources), np.average(MSE_hist)))