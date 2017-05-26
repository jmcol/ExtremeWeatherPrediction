import pandas as pd

class DataLoader(object):

    def load_weather_data ( self ):
        return pd.read_csv( 'hail-2015.csv' )

    def load_climate_data( self ):
        data = pd.read_csv( 'ghcn-m-v1.csv' )
        processed_data = self.preprocess_climate_data( data )
        return processed_data

    def preprocess_climate_data( self, x ):
        years = []
        months = []
        lat_lower_ranges = []
        lat_upper_ranges = []
        lon_lower_ranges = []
        lon_upper_ranges = []
        anomaly_values = []

        x_splice = x[x.year == 2015]

        for index, row in x_splice.iterrows():
            #may be better way to do this than x^2 time
            for col, col_data in row.iteritems():
                if col == 'year' or col == 'month' or col == 'lat':
                    continue

                if col_data != -9999:
                    years.append(row['year'])
                    months.append(row['month'])

                    lower_lat, upper_lat = self.parse_lat_string(row['lat'])
                    lat_lower_ranges.append(lower_lat)
                    lat_upper_ranges.append(upper_lat)

                    lower_lon, upper_lon = self.parse_lon_string(col)
                    lon_lower_ranges.append(lower_lon)
                    lon_upper_ranges.append(upper_lon)

                    anomaly_values.append(col_data / 100.)

        reordered_anomaly_values = {
            'year': pd.Series(years),
            'month': pd.Series(months),
            'lat_lower_range': pd.Series(lat_lower_ranges),
            'lat_upper_range': pd.Series(lat_upper_ranges),
            'lon_lower_range': pd.Series(lon_lower_ranges),
            'lon_upper_range': pd.Series(lon_upper_ranges),
            'anomaly_value': pd.Series(anomaly_values)
        }
        df = pd.DataFrame(reordered_anomaly_values)
        return df

    def parse_lat_string(self, s):
        values = s.split('-')
        lower = int(values[0])
        upper = int(values[1][:-1])
        cardinality = values[1][-1:]

        if cardinality == 'S':
            #multiply values by -1 and switch min and max
            lower, upper = lower * -1, upper * -1
            lower, upper = upper, lower

        return lower, upper

    def parse_lon_string(self, s):
        #TODO: make string values ints
        values = s.split('_')
        lower = int(values[1])
        upper = int(values[2][:-1])
        cardinality = values[2][-1:]

        if cardinality == 'W':
            #multiply values by -1 and switch min and max
            lower, upper = lower * -1 , upper * -1
            lower, upper = upper, lower

        return lower, upper

    def merge_data( self, weather_data, climate_data ):
        #Only allow events with SEVPROB & SEVERE values greater than 0
        weather_data_splice = weather_data[(weather_data.SEVPROB > 0) & (weather_data.PROB > 0)]
        climate_data_splice = climate_data[(climate_data.lon_lower_range > weather_data.LON.min()) &
           (climate_data.lon_upper_range < weather_data.LON.max()) &
           (climate_data.lat_lower_range > weather_data.LON.min()) &
           (climate_data.lat_upper_range < weather_data.LON.max())]

        #intersect weather data with climate data in terms of lat/lon
        #consider time range after climate anomaly
        #want best representation of relationship

        for index, row in climate_data.iterrows():

            # select weather events that occurred within 30 days of anomaly
            monthly_events = weather_data_splice['X.ZTIME']\
                .apply(str).str.startswith('2.015' +
                    str(int(row['month'])).zfill(2))
            monthly_events_df = weather_data_splice[monthly_events]

            # and whose range contain the weather event
            events_in_range = monthly_events_df[(monthly_events_df.LAT > row['lat_lower_range'])
                & (monthly_events_df.LAT < row['lat_upper_range'])
                & (monthly_events_df.LON > row['lon_lower_range'])
                & (monthly_events_df.LON < row['lon_upper_range'])]

            #insert data into row somehow...
            if not events_in_range.empty:
                continue

        return weather_data, climate_data