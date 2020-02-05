import numpy as np
import pickle


def shorten(X, Y):
    """Remove time step exceeding shorter sequence"""
    lens = [np.min([len(x), len(y)]) for x, y in zip(X, Y)]
    X = [x[:l] for x, l in zip(X, lens)]
    Y = [y[:l] for y, l in zip(Y, lens)]
    return X, Y
