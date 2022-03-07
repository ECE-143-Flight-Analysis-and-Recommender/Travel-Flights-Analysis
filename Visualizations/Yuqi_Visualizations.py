import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import pickle
import plotly.graph_objs as go
import plotly.express as px

mpl.rcParams['xtick.color'] = '#FEF4E8'
mpl.rcParams['ytick.color'] = '#FEF4E8'
mpl.rcParams['axes.labelcolor'] = '#FEF4E8'
mpl.rcParams['axes.facecolor'] = '#FEF4E8'
mpl.rcParams['axes.edgecolor'] = '#FEF4E8'
mpl.rcParams['font.size'] = '18'
mpl.rcParams['legend.facecolor'] = 'white'
mpl.rcParams['axes.titlecolor'] = '#FEF4E8'

# sns.set_context("poster")

discrete_cmap = ['#fa8072', '#87cefa', '#f5a700', '#eb471a', '#086569',
                 '#936400', '#ac9e81', '#d60800', '#13e2ea', '#66c2a5', '#d60800', '#e4bb95', '#d5c4a1', '#c8524e']

with open("cmap.dat", "rb") as f:
    cmap = pickle.load(f)

def read_clean_df(flights="/content/drive/MyDrive/flights.csv", airline="/content/drive/MyDrive/airlines.csv", airport="/content/drive/MyDrive/airports.csv"):
    """
    Read data from file, clean it by removing invalid rows and add columns('DELAY', 'ARR_DELAY', 'DEP_DELAY')

    Args:
        flights (str, optional): filepath for flights data. Defaults to "/content/drive/MyDrive/flights.csv".
        airline (str, optional): filepath for flights data. Defaults to "/content/drive/MyDrive/airlines.csv".
        airport (str, optional): filepath for airport data. Defaults to "/content/drive/MyDrive/airports.csv".

    Returns:
        df, df_airline, df_airport(pd.DataFrame): cleaned dataframe
    """
    assert isinstance(flights, str)
    assert isinstance(airline, str)
    assert isinstance(airport, str)

    df = pd.read_csv("/content/drive/MyDrive/flights.csv",
                     dtype={'ORIGIN_AIRPORT': str, 'DESTINATION_AIRPORT': str})
    df = df.drop(df[df['ORIGIN_AIRPORT'].str.isdigit()].index, axis=0)
    tmp = df[df['CANCELLATION_REASON'].isnull()].iloc[:, 0:24]
    df = df.drop(tmp[tmp.isnull().any(axis=1)].index, axis=0)
    df.reset_index(inplace=True, drop=True)
    df['DELAY'] = np.select(
        [df['DEPARTURE_DELAY'] > 0, df['ARRIVAL_DELAY'] > 0], [1, 1], default=0)
    df['ARR_DELAY'] = np.where(df['ARRIVAL_DELAY'] > 0, 1, 0)
    df['DEP_DELAY'] = np.where(df['DEPARTURE_DELAY'] > 0, 1, 0)

    df_airline = pd.read_csv("/content/drive/MyDrive/airlines.csv")

    df_airport = pd.read_csv("/content/drive/MyDrive/airports.csv")
    df_airport = df_airport.drop(
        df_airport[df_airport.isnull().any(axis=1)].index, axis=0)

    df['AIRLINE'] = df['AIRLINE'].map(
        df_airline.set_index('IATA_CODE')['AIRLINE'])

    return df, df_airline, df_airport

df, df_airline, df_airport = read_clean_df(
    "flights.csv", "airlines.csv", "airports.csv")

def plot_cancellation_reason(df, xlabel, ylabel, title):
    """
    plot stacked area plot for reasons of cancellation.

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)
    discrete_cmap = ['#fa8072', '#87cefa', '#f5a700', '#eb471a', '#086569',
                     '#936400', '#ac9e81', '#d60800', '#13e2ea', '#0ea9af', '#d60800', '#e4bb95', '#d5c4a1', '#c8524e']
    # get statistics
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('not cancell')
    df_reason_cancell = pd.DataFrame(df.groupby(['MONTH', 'CANCELLATION_REASON']).size(
    )/df.groupby(['MONTH']).size(), columns=['percent_cancell']).reset_index()

    tmp = pd.DataFrame([[4, 'D', 0], [6, 'D', 0], [9, 'D', 0], [12, 'D', 0], [10, 'A', 0], [10, 'B', 0], [
        10, 'C', 0], [10, 'D', 0]], columns=['MONTH', 'CANCELLATION_REASON', 'percent_cancell'])
    df_reason_cancell = pd.concat(
        [df_reason_cancell, tmp]).reset_index(drop=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.stackplot(range(12), df_reason_cancell[df_reason_cancell['CANCELLATION_REASON'] == 'A']['percent_cancell'],
                 df_reason_cancell[df_reason_cancell['CANCELLATION_REASON']
                                   == 'B']['percent_cancell'],
                 df_reason_cancell[df_reason_cancell['CANCELLATION_REASON']
                                   == 'C']['percent_cancell'],
                 df_reason_cancell[df_reason_cancell['CANCELLATION_REASON']
                                   == 'D']['percent_cancell'],
                 colors=discrete_cmap)

    ax.legend(['Airline/Carrier', 'Weather',
              'National Air System', 'Security'])
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May',
                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Cancellation Rate')
    ax.set_title('Reasons for Cancellation')

    # labels
    ax.legend(['Airline/Carrier', 'Weather',
              'National Air System', 'Security'])
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May',
                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

plot_cancellation_reason(
    df, 'Month', 'Cancellation Rate', 'Reasons for Cancellation')

df = df.drop(df[df['CANCELLED'] == 1].index, axis=0)

def weekday_related_plot1(df, xlabel, ylabel, title):
    """
    plot Flight Counts vs. Weekdays and Month

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel1 (str): x-axis label
        ylabel1 (str): y-axis label
        title1 (str): title of plot

    Return:
        df_dow (pd.DataFrame)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    # get statistics
    df_dow = pd.DataFrame(df[df['DELAY'] == 1].groupby(['MONTH', 'DAY', 'DAY_OF_WEEK']).size(
    )/df.groupby(['MONTH', 'DAY', 'DAY_OF_WEEK']).size(), columns=['delay_rate'])
    df_dow['count'] = df.groupby(['MONTH', 'DAY', 'DAY_OF_WEEK']).size()
    df_dow.reset_index(inplace=True)

    d = {1: 'Mon', 2: 'Tues', 3: 'Wed', 4: 'Thurs', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
    df_dow['day_of_week'] = df_dow['DAY_OF_WEEK'].replace(d)

    # plot
    ax = df_dow.pivot_table('count', index='MONTH', columns='day_of_week',
                            aggfunc=np.mean, sort=True).plot(figsize=(16, 9), color=discrete_cmap)

    # labels
    ax.legend(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri',
              'Sat', 'Sun'], loc="lower right")
    ax.set_xticks(range(13))
    ax.set_xticklabels(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    return df_dow

def weekday_related_plot2(df_dow, xlabel, ylabel, title):
    """
    Delay Rate vs. Weekdays and Month

    Args:
        df_dow (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df_dow, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    # plot
    ax = df_dow.pivot_table('delay_rate', index='MONTH', columns='day_of_week',
                            aggfunc=np.mean, sort=True).plot(figsize=(16, 9), color=discrete_cmap)

    # labels
    ax.legend(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
    ax.set_xticks(range(13))
    ax.set_xticklabels(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

df_dow = weekday_related_plot1(
    df, 'Month', 'Flight Counts', 'Flight Counts vs. Weekdays and Month')

weekday_related_plot2(df_dow, 'Month', 'Delay Rate',
                      'Delay Rate vs. Weekdays and Month')

def plot_arr_dep_delay(df, xlabel, ylabel, title):
    """
    Average Departure Delay vs. Average Arrival Delay

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    # get statistics
    df_arr_dep = pd.DataFrame(df.groupby(
        ['AIRLINE']).size(), columns=['count'])
    df_arr_dep['arr_delay_rate'] = df[df['DELAY'] == 1].groupby(
        ['AIRLINE']).size()/df.groupby(['AIRLINE']).size()
    df_arr_dep['arr_delay'] = df.groupby(['AIRLINE'])['ARRIVAL_DELAY'].mean()
    df_arr_dep['dep_delay'] = df.groupby(['AIRLINE'])['DEPARTURE_DELAY'].mean()

    # plot
    with sns.plotting_context(rc={"legend.fontsize": 13}):
        fig = sns.relplot(x="dep_delay", y="arr_delay", hue="AIRLINE", size="count",
                          sizes=(40, 500), alpha=.5, palette=discrete_cmap,
                          height=6, data=df_arr_dep, edgecolor=None)

        # labels
        fig.fig.set_figheight(10)
        fig.fig.set_figwidth(16)
        fig.set_xlabels(xlabel)
        fig.set_ylabels(ylabel)
        fig.set_titles(title)
        fig._legend.texts[0].set_text('Airline')
        fig._legend.texts[15].set_text('Counts')
        fig._legend.draw_frame(True)
        #fig._legend.set_bbox_to_anchor((-0.2, 0,1,1))
        plt.plot(range(17), range(17), color='#5279a3',
                 linestyle='--', linewidth=2)

plot_arr_dep_delay(df, 'Average Departure Delay', 'Average Arrival Delay',
                   'Average Departure Delay vs. Average Arrival Delay')

def plot_airline_delay(df, xlabel, ylabel):
    """
    Average Departure Delay vs. Average Arrival Delay

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)

    # get statistics
    df['departute_hour'] = np.floor(
        (df['SCHEDULED_DEPARTURE']-1)/100).astype(np.int8)
    df['arrival_hour'] = np.floor(
        (df['SCHEDULED_ARRIVAL']-1)/100).astype(np.int8)

    df_delay_airline = pd.DataFrame(df[df['DELAY'] == 1].groupby(['AIRLINE']).size(
    )/df.groupby(['AIRLINE']).size(), columns=['delay_rate']).reset_index()
    df_delay_airline = df_delay_airline.sort_values(by=['delay_rate'])
    df_delay_airline['higher_than_avg'] = np.where(
        df_delay_airline['delay_rate'] > df_delay_airline['delay_rate'].mean(), 'Higher than average', 'Lower than average')

    # plot
    _, ax = plt.subplots(figsize=(10, 10))
    fig = sns.barplot(x="delay_rate", y="AIRLINE", data=df_delay_airline,
                      hue='higher_than_avg', palette=discrete_cmap[:2][::-1])
    fig.legend(loc="upper right")

    # labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(df_delay_airline['delay_rate'].mean(
    ), color='deepskyblue', label='Mean', linestyle='--', linewidth=2.0)

    return df

df = plot_airline_delay(df, 'Delay Rate', 'Airline')

def plot_heatmap1(df, xlabel, ylabel, title):
    """
    Delays Rate vs. Airline and Arrival Hour

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    df_rate_arr = pd.DataFrame(df[df['DELAY'] == 1].groupby(['arrival_hour', 'AIRLINE']).size(
    )/df.groupby(['arrival_hour', 'AIRLINE']).size(), columns=['delay_rate'])
    df_rate_arr = df_rate_arr.pivot_table(
        'delay_rate', index='AIRLINE', columns='arrival_hour')
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(df_rate_arr[range(6, 24)], cmap=cmap, ax=ax,
                linecolor='black', linewidth=0.01, cbar_kws={'label': 'Delay Rate'})

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

plot_heatmap1(df, 'Arrival Hour', 'Airline',
              'Delays Rate vs. Airline and Arrival Hour')

def plot_heatmap2(df, xlabel, ylabel, title):
    """
    Delays Rate vs. Airline and Departute Hour

    Args:
        df (pd.DataFrame): dataframe of flights
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    df_rate_arr = pd.DataFrame(df[df['DELAY'] == 1].groupby(['departute_hour', 'AIRLINE']).size(
    )/df.groupby(['departute_hour', 'AIRLINE']).size(), columns=['delay_rate'])
    df_rate_arr = df_rate_arr.pivot_table(
        'delay_rate', index='AIRLINE', columns='departute_hour')
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(df_rate_arr[range(6, 24)], cmap=cmap, ax=ax,
                linecolor='black', linewidth=0.01, cbar_kws={'label': 'Delay Rate'})

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

plot_heatmap2(df, 'Departute Hour', 'Airline',
              'Delays Rate vs. Airline and Departute Hour')

def plot_reasons_delay(df, title):
    """
    pie chart for reasons for delay

    Args:
        df (pd.DataFrame): dataframe of flights
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(title, str)

    # get statistics
    df_delay_type = pd.Series([len(df[df['AIR_SYSTEM_DELAY'] > 0]), len(df[df['SECURITY_DELAY'] > 0]),
                               len(df[df['AIRLINE_DELAY'] > 0]), len(
                               df[df['LATE_AIRCRAFT_DELAY'] > 0]),
                               len(df[df['WEATHER_DELAY'] > 0])],
                              ['air_system', 'security', 'airline', 'late_aircraft', 'weather'])

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    patches, texts, pcts = ax.pie(df_delay_type, labels=['Air System', 'Security', 'Airline', 'Late Aircraft', 'Weather'],
                                  autopct='%1.1f%%', colors=discrete_cmap)

    # labels
    for i in range(len(texts)):
        texts[i].set_color('#FEF4E8')

    plt.title(title)
    plt.setp(pcts, color='#FEF4E8')

plot_reasons_delay(df, "Reasons For Delay")

def get_color(cmap, position):
    """
    get RGB color from color map

    Args:
        cmap (mpl.colors.LinearSegmentedColormap): color map
        position (): position on the color map (between 0 and 1)

    Returns:
        str: represents rgb color that can be used in plotly
    """
    assert isinstance(cmap, mpl.colors.LinearSegmentedColormap)
    assert 0 <= position <= 1

    color = cmap(position)
    return f"rgb({int(round(color[0]*255))},{int(round(color[1]*255))},{int(round(color[2]*255))})"

def plot_route_airport(df, df_airport, title):
    """
    plot route and aiport on map as well as delay rate

    Args:
        df (pd.DataFrame): dataframe of flights
        df_airport (pd.DataFrame): dataframe of airport
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_airport, pd.DataFrame)
    assert isinstance(title, str)

    # get statistics
    df['route'] = tuple(zip(df['ORIGIN_AIRPORT'], df['DESTINATION_AIRPORT']))
    df['route'] = df['route'].apply(sorted)
    df['route'] = df['route'].apply(tuple)

    df['latitude_origin'] = df['ORIGIN_AIRPORT'].map(
        df_airport.set_index('IATA_CODE')['LATITUDE'])
    df['latitude_des'] = df['DESTINATION_AIRPORT'].map(
        df_airport.set_index('IATA_CODE')['LATITUDE'])

    df_route = pd.DataFrame(df.groupby(['route']).size(), columns=['count'])
    df_route = df_route[df_route['count'] > 5000].reset_index()
    df_route['origin'], df_route['destination'] = zip(*df_route['route'])
    df_route['latitude_origin'] = df_route['origin'].map(
        df_airport.set_index('IATA_CODE')['LATITUDE'])
    df_route['longtitude_origin'] = df_route['origin'].map(
        df_airport.set_index('IATA_CODE')['LONGITUDE'])
    df_route['latitude_des'] = df_route['destination'].map(
        df_airport.set_index('IATA_CODE')['LATITUDE'])
    df_route['longtitude_des'] = df_route['destination'].map(
        df_airport.set_index('IATA_CODE')['LONGITUDE'])

    df_route['count_delay'] = df_route['route'].map(
        df[df['DELAY'] == 1].groupby(['route']).size())

    df_route['delay_rate'] = df_route['count_delay']/df_route['count']
    df_route = df_route[df_route['delay_rate'] < 0.67].reset_index(drop=True)

    df_airport['delay_rate'] = df_airport['IATA_CODE'].map(df[df['DELAY'] == 1].groupby(
        'ORIGIN_AIRPORT').size()/df.groupby('ORIGIN_AIRPORT').size())
    df_airport['count'] = df_airport['IATA_CODE'].map(
        df.groupby('ORIGIN_AIRPORT').size())
    df_airport = df_airport[df_airport['count'] > 10000].reset_index(drop=True)

    # plot
    c_min, c_max = min(df_route['delay_rate']), max(df_route['delay_rate'])
    fig = px.scatter_mapbox(df_airport, lat="LATITUDE", lon="LONGITUDE", color="delay_rate", size="count", range_color=[c_min, c_max],
                            color_continuous_scale=['#009dff', '#66c0f6', '#cce3ed', '#ffd4c7', '#ff9382', '#ff523e'], size_max=20, zoom=3)

    fig.update_geos(projection_type="orthographic")
    for i in range(len(df_route)):
        p_of_cmap = (df_route['delay_rate'][i]-c_min)/(c_max-c_min)
        fig.add_trace(
            go.Scattermapbox(
                lon=[df_route['longtitude_origin'][i],
                     df_route['longtitude_des'][i]],
                lat=[df_route['latitude_origin'][i],
                     df_route['latitude_des'][i]],
                mode='lines',
                line=dict(width=1, color=get_color(cmap, p_of_cmap)),
                opacity=float(df_route['count'][i]) /
                float(df_route['count'].max()),
                showlegend=False,
            )
        )

    fig.update_layout(mapbox_accesstoken="pk.eyJ1IjoibGFsYWxhMjAyMSIsImEiOiJja3drNGl0djAxb2phMnVubzZkYWhmZ3hkIn0.TbWtnv71s5niti55m-KJ6g",
                      mapbox_style="light",
                      title=title,
                      font={"size": 10, "color": '#FEF4E8'},
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)'
                      )

    fig.update_coloraxes(colorbar_title_text='Delay Rate')

    fig.show()

plot_route_airport(df, df_airport, 'Delay Rate of Flights and Airports in US')

def plot_big_small_airport(df, df_airport, xlabel, ylabel, title):
    """
    Big/Small aiports

    Args:
        df (pd.DataFrame): dataframe of flights
        df_airport (pd.DataFrame): dataframe of airport
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of plot
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_airport, pd.DataFrame)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)

    df_airport['delay_rate'] = df_airport['IATA_CODE'].map(df[df['DELAY'] == 1].groupby(
        'ORIGIN_AIRPORT').size()/df.groupby('ORIGIN_AIRPORT').size())
    df_airport['count'] = df_airport['IATA_CODE'].map(
        df.groupby('ORIGIN_AIRPORT').size())
    df_airport['airport_size'] = np.where(
        df_airport['count'] > 20000, 'Big Airport', 'Small Airport')

    fig = sns.lmplot(x="count", y="delay_rate", fit_reg=False,
                     hue="airport_size", palette=discrete_cmap,
                     data=df_airport)

    fig.fig.set_figheight(10)
    fig.fig.set_figwidth(13)
    fig.legend.set_title("")
    fig.legend.draw_frame(True)
    fig.set_xlabels(xlabel)
    fig.set_ylabels(ylabel)
    fig.set_titles(title)

plot_big_small_airport(df, df_airport, "Flight Counts",
                       "Delay Rate", "Big/Small aiports")
