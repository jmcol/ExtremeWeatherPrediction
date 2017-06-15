from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDClassifier

import pandas as pd
import numpy
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
        ticks = numpy.arange(0, 9, 1)
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
        anomaly_features = self.merged_data.drop(['event_prob', 'event_severe_prob'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(anomaly_features, event_labels,
            test_size=0.25, random_state=42)

        self.regressor.fit(X_train, y_train)
        print "Training score:" + " " + self.regressor.score(X_train, y_train)
        print "Testing score:" + " " + self.regressor.score(X_test, y_test)


if __name__ == '__main__':
    BENCHMARKS = False
    VISUALS = True

    dl = DataLoader()
    reg = MLPRegressor(hidden_layer_sizes=(100, 75),activation='logistic', learning_rate='adaptive', max_iter=10000,
        shuffle=True, random_state=24)

    if BENCHMARKS:
        rng = numpy.random.RandomState(1)

        #Benchmark with default MLP regression
        benchmark_reg1 = MLPRegressor()
        wp_benchmark = WeatherPredictor( benchmark_reg1, dl )
        wp_benchmark.load_data()

        benchmark_reg2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4, ), n_estimators=500, random_state=rng)
        wp_benchmark2 = WeatherPredictor( benchmark_reg2, dl )
        wp_benchmark2.load_data()

        if VISUALS:
            wp_benchmark.visualize_data()

        wp_benchmark.train_predict()
        wp_benchmark2.train_predict()

    clf = SGDClassifier(alpha=0.001, n_iter=100)
    wp = WeatherPredictor( clf, dl )
    wp.load_data()

    if VISUALS:
        wp.visualize_data( wp.merged_data )
        wp.visualize_data( wp.weather_data )

    wp.train_predict()