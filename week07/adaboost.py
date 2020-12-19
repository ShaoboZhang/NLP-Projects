import numpy as np


class Adaboost():
    def __init__(self, max_depth=10, epochs=50):
        self.max_depth = max_depth
        self.epochs = epochs
        self.clfs = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        m = len(x)
        weights = np.ones(m) / m
        y_pred = np.zeros(m)
        for i in range(self.epochs):
            # build a weak classifier
            clf, pred, err = self.__build_single_clf(x, y, weights)
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-16))
            clf['alpha'] = alpha
            self.clfs.append(clf)
            # updata weights
            tmp = -alpha * pred * y
            weights *= np.exp(tmp)
            weights /= weights.sum()
            y_pred += alpha * pred
            err = np.mean(np.sign(y_pred) != y)
            if err == 0:
                break
        return

    def predict(self, x: np.ndarray):
        m = len(x)
        y_pred = np.zeros(m)
        for clf in self.clfs:
            y_pred += clf['alpha'] * self.__classify(x, clf)
        return np.sign(y_pred).astype(np.int8)

    def __build_single_clf(self, x: np.ndarray, y, weights):
        m, n = x.shape
        best_clf = {}
        best_y = np.ones(m)
        min_err = float('inf')
        for dim in range(n):
            min_val = x[:, dim].min()
            max_val = x[:, dim].max()
            step = (max_val - min_val) / self.max_depth
            for thresh in np.arange(min_val, max_val + step, step):
                for sign in ('lt', 'gt'):
                    clf = {'dim': dim, 'thresh': thresh, 'sign': sign}
                    y_pred = self.__classify(x, clf)
                    err = weights @ (y_pred != y)
                    if err < min_err:
                        min_err = err
                        best_clf = clf
                        best_y = y_pred
        return best_clf, best_y, min_err

    @staticmethod
    def __classify(x: np.ndarray, clf: dict):
        m, n = x.shape
        preds = np.ones(m)
        dim = clf['dim']
        thresh = clf['thresh']
        if clf['sign'] == 'lt':
            preds[x[:, dim] < thresh] = -1
        else:
            preds[x[:, dim] > thresh] = -1
        return preds


if __name__ == '__main__':
    x = np.array([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]])
    y = np.array([1, 1, -1, -1, 1])
    ada_clf = Adaboost(10, 10)
    ada_clf.fit(x, y)
    print(ada_clf.predict(x))
