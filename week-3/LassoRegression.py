#!/usr/bin/env python3

import matplotlib.pyplot as plt
from load import load_dataset

if __name__ == '__main__':
    data = load_dataset()
    x, y = data[:, [0, 1]], data[:, 2]
    print(x.shape)

    ax = plt.axes(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], y)

    plt.show()
