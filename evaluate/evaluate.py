from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Evaluator:

    def __init__(self, model, X_train, y_train, X_test, y_test, name=None):
        self._model = model
        self._name = name
        if self._name is None:
            self._name = self._model.__class__.__name__

        # datasets
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        # backup
        self._backup = {
            'X_train': self._X_train,
            'y_train': self._y_train,
            'X_test': self._X_test,
            'y_test': self._y_test
        }

    def preprocess(self, fn_train=None, fn_test=None):
        if fn_train:
            self._X_train, self._y_train = fn_train(self._X_train, self._y_train)
        if fn_test:
            self._X_test, self._y_test = fn_test(self._X_test, self._y_test)
        return self

    def reset(self):
        self._X_train = self._backup['X_train']
        self._y_train = self._backup['y_train']
        self._X_test = self._backup['X_test']
        self._y_test = self._backup['y_test']
        return self

    def visualize(self, fn):
        fn(self._model, self._X_train, self._y_train, self._X_test, self._y_test)

    def evaluate(self):
        self._model.fit(self._X_train, self._y_train)
        pred = self._model.predict(self._X_test)
        mse = mean_squared_error(self._y_test, pred)
        mae = mean_absolute_error(self._y_test, pred)
        r2 = r2_score(self._y_test, pred)
        return {'mse': mse, 'r2': r2, 'mae': mae}

