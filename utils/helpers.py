import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_hist(hist, keys, name, **kwargs):
    for k in keys:
        assert k in hist.keys(), "keys must be in hist"
    if 'plot' in kwargs.keys():
        fig = plt.figure(figsize=(8, 6))
        lines = [plt.plot(range(1, len(hist[k])+1), hist[k], label=k)[0] for k in keys]
        plt.legend(handles=lines)
        plt.grid(True)
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('saved_model/hists/' + name + '_curve.png')
    if 'csv' in kwargs.keys():
        df = pd.DataFrame(hist)
        df.to_csv('saved_model/hists/' + name + '_log.csv')


def one_hot(num, length=11):
    one_hot = [0] * length
    one_hot[num] = 1
    return np.array(one_hot)