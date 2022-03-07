import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import seaborn as sns
import plotly.graph_objects as go

import os

airports = pd.read_csv('airports.csv')
flights = pd.read_csv('flights.csv')
columns_to_drop = ['CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
                  "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
flights = flights.drop(columns_to_drop, axis=1, errors='ignore')
flights = flights[flights['CANCELLED'] == 0]
flights = flights[flights['DIVERTED'] == 0]
corr_matrix = flights.corr()
corr_matrix['ARRIVAL_DELAY'].sort_values(ascending=False)
flights.isnull().sum()

def delay_by_attribute(attribute, df=flights, figsize=(10, 7)):
    # Delay with less than 10 min are mapped to 0 otherwise they are mapped to 1
    delay_type = lambda x: 0 if x < 10 else 1
    flights['DELAY_TYPE'] = flights['DEPARTURE_DELAY'].apply(delay_type)


    plt.figure(1, figsize=figsize)
    ax = sns.countplot(y=attribute, hue='DELAY_TYPE', data=df, palette="Set2")

    plt.xlabel('Flight count', fontsize=16, weight='bold')
    plt.ylabel(attribute, fontsize=16, weight='bold')
    plt.title(f'Delay by {attribute}', weight='bold')
    L = plt.legend()
    L.get_texts()[0].set_text('small delay (t < 10 min)')
    L.get_texts()[1].set_text('large delay (t > 10 min)')
    plt.grid(True)
    plt.show()
delay_by_attribute('AIRLINE')
result = pd.merge(flights[['ORIGIN_AIRPORT', 'DELAY_TYPE']], airports[['IATA_CODE', 'STATE']], left_on='ORIGIN_AIRPORT', right_on='IATA_CODE')

delay_by_attribute('STATE', df=result, figsize=(10, 15))

airports = pd.read_csv("airports.csv", delimiter=',')
flights = pd.read_csv("flights.csv", delimiter=',')
flights['delay'] = flights['DEPARTURE_DELAY'] + flights['ARRIVAL_DELAY']
flights_clear = flights.drop(flights[flights['delay'] <= 0].index)
source_airports = airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]
destination_airports = source_airports.copy()
source_airports.columns = [str(col) + '_source' for col in source_airports.columns]
destination_airports.columns = [str(col) + '_destination' for col in destination_airports.columns]
routes = flights_clear[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']]
routes = pd.merge(routes, source_airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE_source')
routes = pd.merge(routes, destination_airports, left_on='DESTINATION_AIRPORT', right_on='IATA_CODE_destination')
fig = go.Figure()
fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = airports['LONGITUDE'],
    lat = airports['LATITUDE'],
    hoverinfo = 'text',
    text = airports['AIRPORT'],
    mode = 'markers',
    marker = dict(
        size = 2,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))
flight_paths = []
for i in range(100):
    fig.add_trace(
        go.Scattergeo(
            locationmode = 'USA-states',
            lon = [routes['LONGITUDE_source'][i], routes['LONGITUDE_destination'][i]],
            lat = [routes['LATITUDE_source'][i], routes['LATITUDE_destination'][i]],
            mode = 'lines',
            line = dict(width = 0.1,color = 'red'),
        )
    )
fig.update_layout(
    title_text = '2015 delay flight paths',
    showlegend = False,
    geo = dict(
        scope = 'north america',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)
fig.show()
