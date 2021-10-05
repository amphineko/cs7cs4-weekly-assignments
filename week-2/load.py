import numpy as np


def load_dataset():
    return np.genfromtxt('data.csv', delimiter=',', dtype=float, skip_header=True)


if __name__ == '__main__':
    data = load_dataset()
    print(data.head())
