#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from load import load_dataset

polyFx = PolynomialFeatures(degree=5)

if __name__ == '__main__':
    data = load_dataset()
    x, y = data[:, :-1], data[:, -1]

    # create additional polynomial features
    x = polyFx.fit_transform(x, y)

    # split train and test dataset
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

    # create and fit the Lasso model
    lasso = Lasso(alpha=0.025)
    lasso.fit(x, y)

    # score the model
    score = lasso.score(x_test, y_test)
    print('Score: ', score)
