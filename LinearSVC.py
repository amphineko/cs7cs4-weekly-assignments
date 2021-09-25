import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from load import load_dataset

if __name__ == '__main__':
    plt.set_cmap('Set2')

    # load dataset
    data = load_dataset()
    np.random.shuffle(data)

    x_all, y_all = data[:, :2], data[:, 2].astype('int')
    part = int(0.9 * len(y_all))
    x1, x2 = x_all[:, 0], x_all[:, 1]

    x, y = x_all[:part], y_all[:part]  # training dataset
    x_test, y_test = x_all[part:], y_all[part:]  # test dataset

    # fit the model with training dataset
    svc = SVC(kernel='linear')
    svc.fit(x, y)

    # predict with test dataset
    y_r = svc.predict(x_test)

    # configure plotting boundary
    framing = 0.25
    x1_min, x1_max = x1.min() - framing, x1.max() + framing
    x2_min, x2_max = x2.min() - framing, x2.max() + framing
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # plot training points
    plt.scatter(*x[y == -1].T, c='tab:orange', marker='1')
    plt.scatter(*x[y == +1].T, c='tab:blue', marker='2')

    # plot test points
    plt.scatter(*x_test[y_r == y_test].T, c='tab:green', marker='+')
    plt.scatter(*x_test[y_r != y_test].T, c='tab:red', marker='x')

    # plot decision boundary
    xx1, xx2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    z = svc.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    plt.contour(xx1, xx2, z.reshape(xx1.shape),
                colors=['tab:gray', 'tab:red', 'tab:gray'],
                levels=[-.5, 0, .5],
                linestyles=['--', '-', '--'],)

    plt.legend(['train: -1',
                'train: +1',
                'test: ok',
                'test: error',
                ])
    plt.title('LinearSVC')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig('LinearSVC')
    plt.show()
