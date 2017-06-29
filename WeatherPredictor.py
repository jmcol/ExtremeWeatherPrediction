from sklearn.model_selection import train_test_split
from DataLoader import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class WeatherPredictor( object ):

    def __init__( self, regressor, data_loader ):
        self.regressor = regressor
        self.data_loader = data_loader

    def visualize_data( self, data ):
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
        
    def load_data( self ):
        self.climate_data = self.data_loader.load_climate_data()
        self.weather_data = self.data_loader.load_weather_data()
        self.merged_data = self.data_loader.merge_data(self.weather_data, self.climate_data)
        self.merged_data = self.data_loader.preprocess_merged_data( self.merged_data )

    def train_predict( self ):
        event_labels = self.merged_data[['event_prob']]
        anomaly_features = self.merged_data.drop(['event_prob'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(anomaly_features, event_labels,
            test_size=0.25, random_state=42)

        robust_scaler = RobustScaler()
        Xtr_r = robust_scaler.fit_transform(X_train)
        Xte_r = robust_scaler.transform(X_test)

        self.regressor.fit(Xtr_r, y_train.values.ravel())
        print "Training score:" + " " + self.regressor.score(Xtr_r, y_train)
        print "Testing score:" + " " + self.regressor.score(Xte_r, y_test)


if __name__ == '__main__':
    BENCHMARKS = False
    VISUALS = True

    dl = DataLoader()

    if BENCHMARKS:
        rng = np.random.RandomState(1)

        #Benchmark with default MLP regression
        benchmark_reg1 = LinearSVC()
        wp_benchmark = WeatherPredictor( benchmark_reg1, dl )
        wp_benchmark.load_data()

        if VISUALS:
            wp_benchmark.visualize_data()

        wp_benchmark.train_predict()

    tuned_parameters = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                         'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                         'alpha': 10.0**-np.arange(1,7)}]

    clf = SGDClassifier(alpha=0.0001,loss='log',penalty='none',fit_intercept=True,
                        shuffle=True,random_state=42,n_jobs=-1,n_iter=2)
    wp = WeatherPredictor( clf, dl )
    wp.load_data()

    if VISUALS:
        wp.visualize_data( wp.merged_data )
        wp.visualize_data( wp.weather_data )

    wp.train_predict()