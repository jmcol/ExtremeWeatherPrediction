from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader

class WeatherPredictor( object ):

    def __init__( self, regressor, data_loader ):
        self.regressor = regressor
        self.data_loader = data_loader

    def train_predict( self ):
        climate_data = self.data_loader.load_climate_data()
        weather_data = self.data_loader.load_weather_data()
        merged_data = self.data_loader.merge_data(weather_data, climate_data)

        merged_data.drop(merged_data.columns[0], axis=1, inplace=True)
        anomaly_features = merged_data[['anomaly_month', 'anomaly_value']]
        event_labels = merged_data[['event_prob', 'event_severe_prob', 'event_max_size']]

        X_train, X_test, y_train, y_test = train_test_split(anomaly_features, event_labels,
            test_size=0.25, random_state=42)

        self.regressor.fit(X_train, y_train)
        print "Training score:" + " " + self.regressor.score(X_train, y_train)
        print "Testing score:" + " " + self.regressor.score(X_test, y_test)


if __name__ == '__main__':
    dl = DataLoader()
    reg = MLPRegressor()
    wp = WeatherPredictor( reg, dl )
    wp.train_predict()