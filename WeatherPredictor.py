import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

from DataLoader import DataLoader

params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]


def plot_on_dataset(x, y, name='Weather Data'):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)

    x = MinMaxScaler().fit_transform(x)

    mlp = MLPClassifier(verbose=0, random_state=0,
                        max_iter=15, solver='adam',
                        learning_rate_init=0.01)

    # some parameter combinations will not converge as can be seen on the9
    # plots so they are ignored here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(x, y.values.ravel())

    print("Training set score: %f" % mlp.score(x, y))
    print("Training set loss: %f" % mlp.loss_)
    print("Training set layers: %f" % mlp.n_layers_)
    print("Training set hidden layer sizes: %f" % mlp.hidden_layer_sizes)


class WeatherPredictor(object):

    def __init__(self, regressor, data_loader):
        self.regressor = regressor
        self.data_loader = data_loader

        self.climate_data = None
        self.weather_data = None
        self.merged_data = None
        self.merged_data_sgd = None
        self.merged_data_mlp = None

    @staticmethod
    def visualize_data(data):
        data.hist()
        plt.show()

        pd.scatter_matrix(data.sample(n=1000))
        plt.show()

        data1 = data.sample(n=1000)
        names = list(data.columns.values)
        correlations = data1.corr()
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.show()
        
    def load_data(self):
        self.climate_data = self.data_loader.load_climate_data()
        self.weather_data = self.data_loader.load_weather_data()
        self.merged_data = self.data_loader.merge_data(self.weather_data, self.climate_data)
        self.merged_data_sgd = self.data_loader.preprocess_merged_data(self.merged_data)
        # self.merged_data_mlp = self.data_loader.preprocess_merged_data_mlp(self.merged_data)

    def train_predict(self):
        event_labels = self.merged_data_sgd[['event_prob']]
        anomaly_features = self.merged_data_sgd.drop(['event_prob'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(anomaly_features,
                                                            event_labels,
                                                            test_size=0.25, 
                                                            random_state=42)

        robust_scaler = RobustScaler()
        xtr_r = robust_scaler.fit_transform(x_train)
        xte_r = robust_scaler.transform(x_test)
        self.regressor.fit(xtr_r, y_train.values.ravel())

        print("Training score:" + " " + str(self.regressor.score(xtr_r, y_train)))
        print("Testing score:" + " " + str(self.regressor.score(xte_r, y_test)))

    def train_predict_mlp(self):
        event_labels = self.merged_data_mlp[['event_prob']]
        anomaly_features = self.merged_data_mlp.drop(['event_prob'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(anomaly_features,
                                                            event_labels,
                                                            test_size=0.25,
                                                            random_state=42)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_tr = scaler.transform(x_train)
        x_te = scaler.transform(x_test)

        mlp = MLPRegressor(verbose=0, random_state=0,
                           max_iter=15, solver='adam',
                           learning_rate_init=0.01)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(x_tr, y_train.values.ravel())

        print("Training set score: %f" % mlp.score(x_tr, y_train))
        print("Testing set score: %f" % mlp.score(x_te, y_test))
        print("Training set loss: %f" % mlp.loss_)
        print("Training set layers: %f" % mlp.n_layers_)
        print("Training set hidden layer sizes: %f" % mlp.hidden_layer_sizes)


if __name__ == '__main__':
    BENCHMARKS = False
    VISUALS = False

    dl = DataLoader()

    if BENCHMARKS:
        rng = np.random.RandomState(1)

        # Benchmark with default MLP regression
        benchmark_reg1 = LinearSVC()
        wp_benchmark = WeatherPredictor(benchmark_reg1, dl)
        wp_benchmark.load_data()

        if VISUALS:
            wp_benchmark.visualize_data()

        wp_benchmark.train_predict()

    tuned_parameters = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                         'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                         'alpha': 10.0**-np.arange(1, 7)}]

    clf = SGDClassifier(alpha=0.0001, loss='log', penalty='none', fit_intercept=True,
                        shuffle=True, random_state=42, n_jobs=-1, n_iter=5)
    wp = WeatherPredictor(clf, dl)
    wp.load_data()

    if VISUALS:
        wp.visualize_data(wp.merged_data_sgd)
        wp.visualize_data(wp.weather_data)

    wp.train_predict()
    # wp.train_predict_mlp()
