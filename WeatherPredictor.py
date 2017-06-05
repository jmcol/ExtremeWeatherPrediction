from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader
from pandas.tools.plotting import scatter_matrix

import numpy
import matplotlib.pyplot as plt

class WeatherPredictor( object ):

    def __init__( self, regressor, data_loader ):
        self.regressor = regressor
        self.data_loader = data_loader

    def visualize_data( self ):
        self.merged_data.hist()
        plt.show()

        scatter_matrix(self.merged_data.sample(n=1000))
        plt.show()

        data = self.merged_data.sample(n=1000)
        names = list(self.merged_data.columns.values)
        correlations = data.corr()
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
        climate_data = self.data_loader.load_climate_data()
        weather_data = self.data_loader.load_weather_data()
        self.merged_data = self.data_loader.merge_data(weather_data, climate_data)
        self.merged_data.drop(self.merged_data.columns[0], axis=1, inplace=True)

    def train_predict( self ):
        anomaly_features = self.merged_data[['anomaly_month', 'anomaly_value', 'event_lat', 'event_lon']]
        event_labels = self.merged_data[['event_prob']]

        X_train, X_test, y_train, y_test = train_test_split(anomaly_features, event_labels,
            test_size=0.25, random_state=42)

        self.regressor.fit(X_train, y_train.values.ravel())
        print "Training score:" + " " + self.regressor.score(X_train, y_train)
        print "Testing score:" + " " + self.regressor.score(X_test, y_test)


if __name__ == '__main__':
    BENCHMARKS = False

    dl = DataLoader()
    reg = MLPRegressor(hidden_layer_sizes=(100, 75),activation='logistic', learning_rate='adaptive', max_iter=5000,
        shuffle=True, random_state=24)

    if BENCHMARKS:
        #Benchmark with default MLP regression
        benchmark_reg = MLPRegressor()
        wp_benchmark = WeatherPredictor( benchmark_reg, dl )
        wp_benchmark.load_data()
        wp_benchmark.visualize_data()
        wp_benchmark.train_predict()

    wp = WeatherPredictor( reg, dl )
    wp.load_data()
    wp.visualize_data()
    wp.train_predict()