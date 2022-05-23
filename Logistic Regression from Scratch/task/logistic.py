# write your code here
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.e**(-t))

    def predict_proba(self, row, coef_):
        t = row @ coef_[1:]
        if self.fit_intercept:
            t += coef_[0]
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = np.zeros(X_train.shape[1] + 1)  # initialized weights
        for _ in range(self.n_epoch):
            for i in range(X_train.shape[0]):
                row = X_train.iloc[i, :]
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train.iloc[i]) * y_hat * (1 - y_hat)
                for j in range(len(self.coef_) - 1):
                    self.coef_[j + 1] = self.coef_[j + 1] - self.l_rate * (y_hat - y_train.iloc[i]) * y_hat * (1 - y_hat) * X_train.iloc[i][j]

    def fit_log_loss(self, X_train, y_train):
        number_of_rows = X_train.shape[0]
        self.coef_ = np.zeros(X_train.shape[1] + 1)  # initialized weights
        for _ in range(self.n_epoch):
            for i in range(number_of_rows):
                row = X_train.iloc[i, :]
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train.iloc[i]) / number_of_rows
                for j in range(len(self.coef_) - 1):
                    self.coef_[j + 1] = self.coef_[j + 1] - self.l_rate * (y_hat - y_train.iloc[i]) * X_train.iloc[i][j] / number_of_rows

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for i in range(X_test.shape[0]):
            row = X_test.iloc[i, :]
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat < cut_off:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions # predictions are binary values - 0 or 1

    def standardize(self, X):
        size = X.shape[1]
        for num in range(size):
            col = X.iloc[:, num]
            mean = col.mean()
            sd = col.std()
            X.iloc[:, num] = X.iloc[:, num].map(lambda a: (a - mean)/sd)


def main():
    data = load_breast_cancer(as_frame=True)["frame"]
    X = data.loc[:, ['worst concave points', 'worst perimeter', 'worst radius']]
    y = data["target"]

    regr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    regr.standardize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

    regr.fit_log_loss(X_train, y_train)
    prediction = regr.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    dictionary = dict(coef_=list(regr.coef_), accuracy=accuracy)
    print(dictionary)


if __name__ == '__main__':
    main()
