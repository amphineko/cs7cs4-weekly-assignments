import numpy as np
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier

from load import load_dataset

if __name__ == '__main__':
    plt.set_cmap('Set1')

    # load data
    data = load_dataset()
    np.random.shuffle(data)

    x_all, y_all = data[:, :2], data[:, 2].astype('int')
    part = int(0.9 * len(y_all))
    x1, x2 = x_all[:, 0], x_all[:, 1]

    x, y = x_all[:part], y_all[:part]  # training dataset
    x_test, y_test = x_all[part:], y_all[part:]  # test dataset

    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(x, y)

    print('score: ', clf.score(x_test, y_test))
