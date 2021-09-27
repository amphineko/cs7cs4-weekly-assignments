import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

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
    x = np.hstack((x, x ** 2))  # add square to each feature

    x_test, y_test = x_all[part:], y_all[part:]  # test dataset

    # fit the model with training dataset
    logreg = LogisticRegression(C=1e5)
    logreg.fit(x_all, y_all)
    incpt = logreg.intercept_[0]
    w1, w2 = logreg.coef_.T
    print(logreg.coef_.T)

    # predict with test dataset
    y_r = logreg.predict(x_test)

    # score
    print('score: ', logreg.score(x_test, y_test))

    # configure plotting boundary
    framing = 0.25
    x1_min, x1_max = x1.min() - framing, x1.max() + framing
    x2_min, x2_max = x2.min() - framing, x2.max() + framing
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # plot decision boundary
    offset = -incpt / w2
    m = -w1 / w2
    line_x = np.array([x1_min, x1_max])
    line_y = m * line_x + offset
    plt.plot(line_x, line_y, 'r', lw=1, ls='-')

    # plot training points
    plt.scatter(*(x[:, :2][y == -1]).T, c='tab:orange', marker='1')
    plt.scatter(*(x[:, :2][y == +1]).T, c='tab:blue', marker='2')

    # plot test points
    plt.scatter(*x_test[y_r == y_test].T, c='tab:green', marker='+')
    plt.scatter(*x_test[y_r != y_test].T, c='tab:red', marker='x')

    xx1, xx2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    z = logreg.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    plt.contour(xx1, xx2, z.reshape(xx1.shape),
                colors=['tab:red'],
                levels=[0],
                linestyles=['--', '-', '--'],)

    plt.legend(['decision',
                'train: -1',
                'train: +1',
                'test: ok',
                'test: error',
                ])
    plt.title('LogisticRegression^2')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig('LogisticRegression')
    plt.show()
