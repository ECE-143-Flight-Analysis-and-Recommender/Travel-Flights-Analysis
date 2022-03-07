import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as graph
import calmap
import squarify

from datetime import datetime
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap

def prepare_df(fname):
  '''
  Reads data from flights.csv, removes empty/contradicting values, and then adds
  new columns (datetime objects, combined and separate delays).

  :param fname: filename (e.g. 'flights.csv')
  :type fname: str

  Returns:
  cleaned dataframe with new columns as described above.
  '''
  assert isinstance(fname, str)

  df = pd.read_csv(fname)                                                         # Read in data
  df = df[df.ORIGIN_AIRPORT.apply(lambda val: not isinstance(val, int))]          # Remove non-int values
  df = df[df.DESTINATION_AIRPORT.apply(lambda val: not isinstance(val, int))]
  df = df.iloc[:, 0:24]                                                           # Only get first 24 cols as last few cols are very sparse
  df.dropna(inplace=True)                                                         # Drop NaN
  df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
  df['DELAY'] = np.select([df['DEPARTURE_DELAY'] > 0, df['ARRIVAL_DELAY'] > 0], [1, 1], default=0)
  df['ARR_DELAY'] = np.where(df['ARRIVAL_DELAY'] > 0, 1, 0)
  df['DEP_DELAY'] = np.where(df['DEPARTURE_DELAY'] > 0, 1, 0)
  return df

df = prepare_df('flights.csv')
print(df)

def plot_hist(col, xlabel, ylabel, title):
  '''
  Plot histogram of df column (used for delay columns) with column mean highlighted.

  :param col: column of dataframe df
  :type col: pd.Series
  :param xlabel: x-axis label
  :type xlabel: str
  :param ylabel: y-axis label
  :type ylabel: str
  :param title: title of plot
  :type title: str

  Returns:
  Plot showing histogram and mean of dataframe column.
  '''
  assert isinstance(col, pd.Series) and all(col.apply(float.is_integer))
  assert all([isinstance(x, str) for x in [xlabel, ylabel, title]])

  # Get statistics for column
  mean = col.mean()
  std = col.std()
  total = col.value_counts().to_dict()
  min_thresh = int(mean - 2*std)
  max_thresh = int(mean + 2*std)

  # Settings
  plt.rcParams['font.size'] = '18'
  with plt.rc_context({'axes.edgecolor':'#FEF4E8',
                       'xtick.color':'#FEF4E8',
                       'ytick.color':'#FEF4E8',}):
    plt.figure(figsize=(8,6))
    ax = plt.axes()
    ax.set_facecolor('#FEF4E8')

    # Plot
    plt.hist(col, color='salmon',density=True, bins=list(range(min_thresh, max_thresh)))
    plt.axvline(mean, color='deepskyblue', label='Mean', linestyle='--', linewidth=3.0)
    plt.legend(loc="upper left")
    min, max = plt.gca().get_ylim()
    plt.text(mean+5, max/2, f'{mean:.2f} minutes', color='deepskyblue')

    # Labels
    plt.xlabel(xlabel, color='#FEF4E8')
    plt.ylabel(ylabel, color='#FEF4E8')
    plt.title(title, color='#FEF4E8')


    plt.show()

def plot_del_by(del_col, col, xlabel, ylabel, title, df=df, top=20, figsize=(10,8)):
  '''
  Plot delay percentages calculated in del_col by another column col in a stacked bar plot.

  :param del_col: delay column of dataframe df
  :type del_col: pd.Series
  :param col: name of column of dataframe df that is being organized against in the plot
  :type col: str
  :param xlabel: x-axis label
  :type xlabel: str
  :param ylabel: y-axis label
  :type ylabel: str
  :param title: title of plot
  :type title: str
  :param df: dataframe
  :type df: pd.DataFrame
  :param top: parameter to show only `top` columns
  :type top: int
  :param figsize: tuple indicating plot figsize
  :type figsize: tuple

  Returns:
  Plot showing stacked bar plot of delay percentages.
  '''
  assert isinstance(del_col, pd.Series) and isinstance(col, str) and isinstance(df, pd.DataFrame)
  assert all([isinstance(x, str) for x in [xlabel, ylabel, title]])
  assert isinstance(top, int) and top > 0
  assert isinstance(figsize, tuple) and len(figsize) == 2 and all([dim > 0 for dim in list(figsize)])

  # Organize delay data into form needed for stacked bar plots (lists)
  delays = df.loc[del_col == 1]
  unnorm_del = list(delays[col].value_counts().sort_index())[:top]
  no_delays = df.loc[del_col == 0]
  unnorm_no_del = list(no_delays[col].value_counts().sort_index())[:top]

  count_dict = dict(df[col].value_counts().sort_index())
  keys = list(count_dict.keys())[:top]
  vals = list(count_dict.values())[:top]

  del_norm = [i/j for i,j in zip(unnorm_del, vals)]
  no_del_norm = [i/j for i,j in zip(unnorm_no_del, vals)]

  # Plot parameters
  counts = [del_norm, no_del_norm]
  labels = ['DELAY', 'NO DELAY']
  colors = ['salmon', 'goldenrod']
  xax = np.arange(len(keys))
  n = len(counts)
  width = 1.5

  # Settings
  plt.rcParams['font.size'] = '18'
  with plt.rc_context({'axes.edgecolor':'#FEF4E8',
                       'xtick.color':'#FEF4E8',
                       'ytick.color':'#FEF4E8'}):
    fig, ax = plt.subplots(figsize=figsize)
    axe = plt.axes()
    axe.set_facecolor('#FEF4E8')

    # Update for each stacked bar
    bot = [0]*len(keys)
    for i in range(n):
      curr_c = counts[i]
      curr_l = labels[i]
      curr_col = colors[i]
      plt.bar(xax, curr_c, width/n, label=curr_l, bottom=bot, color=curr_col)
      bot = [x+y for x,y in zip(bot, curr_c)]

    plt.xlabel(xlabel, color='#FEF4E8')
    plt.ylabel(ylabel, color='#FEF4E8')
    plt.title(title, color='#FEF4E8')
    plt.xticks(xax, keys)
    plt.legend()
    plt.show()

def plot_by_state(title, df, airport_fname):
  '''
  Plot USA choropleth map to show delays per state.

  :param title: title of plot
  :type title: str
  :param df: dataframe
  :type df: pd.DataFrame
  :param airport_fname: filename of airport csv file (airports.csv)
  :type airport_fname: str

  Returns:
  Choropleth plot showing delays per state
  '''
  assert isinstance(title, str) and isinstance(airport_fname, str)
  assert isinstance(df, pd.DataFrame)

  # Get state data to dataframe
  airport_df = pd.read_csv(airport_fname)
  state_dict = airport_df.set_index('IATA_CODE').to_dict()['STATE']
  df['ORIGIN_STATE'] = df['ORIGIN_AIRPORT'].map(state_dict)

  # Create new dataframe with state and associated densities
  delay = df.loc[df.DELAY==1]
  counts_dict = {'State': delay.ORIGIN_STATE.value_counts().index.to_list(),
                'Density': delay.ORIGIN_STATE.value_counts().to_list()
                }
  state_to_counts = pd.DataFrame.from_dict(counts_dict)
  total = state_to_counts['Density'].sum()
  state_to_counts['Density'] = state_to_counts['Density'].apply(lambda x: (100*x/total))

  # Plot choropleth using plotly
  map_plot = graph.Figure(
    data=graph.Choropleth(
      locations=state_to_counts['State'],
      z = state_to_counts['Density'],
      locationmode = 'USA-states',
      colorscale = 'YlOrBr',
      marker_line_color='white',
      colorbar_title = "Delay rate"),
    layout= graph.Layout(
        geo_scope='usa',
        title = title,
        font = {"size": 10, "color":'#FEF4E8'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    ))
  map_plot.show()

def plot_by_date(xlabel, ylabel, title, df):
  '''
  Plots sliding window time series of dates and shows the densities of delays for each date.

  :param xlabel: x-axis label
  :type xlabel: str
  :param ylabel: y-axis label
  :type ylabel: str
  :param title: title of plot
  :type title: str
  :param df: dataframe
  :type df: pd.DataFrame

  Returns:
  Time series plot showing delays correlated with dates.
  '''
  assert all([isinstance(x, str) for x in [xlabel, ylabel, title]])
  assert isinstance(df, pd.DataFrame)

  # Get delay densities per datetime and approx. over sliding window
  delay = df.loc[df.DELAY==1]
  counts_dict = {'Datetime': delay.DATETIME.value_counts().index.to_list(),
                'Num': delay.DATETIME.value_counts().to_list()
                }
  dates_to_counts = pd.DataFrame.from_dict(counts_dict)
  dates_to_counts = dates_to_counts.sort_values(by=['Datetime'])
  dates_to_counts['Sliding'] = dates_to_counts['Num'].rolling(14, min_periods=1).mean()

  # Convert to percentages
  total = dates_to_counts['Sliding'].sum()
  dates_to_counts['Sliding'] = dates_to_counts['Sliding'].apply(lambda x: 100*x/total)

  # Settings
  plt.rcParams['font.size'] = '18'
  with plt.rc_context({'axes.edgecolor':'#FEF4E8',
                        'xtick.color':'#FEF4E8',
                        'ytick.color':'#FEF4E8'}):

    # Plot actual datetime data
    plt.figure(figsize=(18,5))
    markers = [datetime(2015, 1, 1), datetime(2015, 5, 23), datetime(2015, 7, 3), datetime(2015, 9, 7), datetime(2015, 11, 26), datetime(2015, 12, 25)]
    with sns.axes_style(rc={'axes.facecolor':'#FEF4E8'}):
      sns.lineplot(x='Datetime', y='Sliding', data=dates_to_counts, label='Delay rate', color='deepskyblue')

    # Mark holidays and spans of vacations (march break, summer break, winter break)
    for date in markers:
      plt.axvline(date, color='deepskyblue', linewidth=2.0, linestyle='--')
    plt.axvspan(datetime(2015, 3, 1), datetime(2015, 3, 21), color='yellowgreen', alpha=0.4, lw=0)
    plt.axvspan(datetime(2015, 6, 22), datetime(2015, 8, 31), color='orangered', alpha=0.4, lw=0)
    plt.axvspan(datetime(2015, 12, 15), datetime(2015, 12, 31), color='deepskyblue', alpha=0.4, lw=0)
    plt.axvspan(datetime(2015, 1, 1), datetime(2015, 1, 4), color='deepskyblue', alpha=0.4, lw=0)

    plt.xlabel(xlabel, color='#FEF4E8')
    plt.ylabel(ylabel, color='#FEF4E8')
    plt.title(title, color='#FEF4E8')

    plt.show()

def plot_calmap(df):
  '''
  Plots calender map of delay instances.

  :param df: dataframe
  :type df: pd.DataFrame

  Returns:
  Calender map as described above.
  '''
  assert isinstance(df, pd.DataFrame)

  # Convert counts to Series.
  delay = df.loc[df.DELAY==1]
  counts_dict = {'Datetime': delay.DATETIME.value_counts().index.to_list(),
                'Num': delay.DATETIME.value_counts().to_list()
                }
  dates_to_counts = pd.DataFrame.from_dict(counts_dict)
  dates_to_counts = dates_to_counts.sort_values(by=['Datetime'])
  dates_to_counts = dates_to_counts.set_index('Datetime').squeeze()

  # Plot calender map.
  plt.figure(figsize=(15,15))
  ax=calmap.yearplot(dates_to_counts, year=2015, cmap='Wistia')
  ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ha='center', color='#FEF4E8')
  ax.set_yticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], rotation='horizontal', va='center', color='#FEF4E8')
  plt.show()

def plot_square(df, month=True):
  '''
  Plots square map of delays for each month or weekday.

  :param df: dataframe
  :type df: pd.DataFrame
  :param month: flag for plotting month or weekday
  :type month: bool

  Returns:
  Plots square map showing proportion of delays.
  '''
  assert isinstance(df, pd.DataFrame)
  assert isinstance(month, bool)

  delay = df.loc[df.DELAY==1]
  if month:
    counts = delay.MONTH.value_counts().sort_index().to_list()
    total = sum(counts)
    counts = [100*x/total for x in counts]
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    labels = [labels[i] + '\n' + f'{counts[i]:.2f}' + '%' for i in range(12)]
  else:
    counts = delay.DAY_OF_WEEK.value_counts().sort_index().to_list()
    total = sum(counts)
    counts = [100*x/total for x in counts]
    labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    labels = [labels[i] + '\n' + f'{counts[i]:.2f}' + '%' for i in range(7)]

  cmap = plt.get_cmap('autumn')
  plt.figure(figsize=(12,6))
  squarify.plot(counts, color=cmap(counts/(0.5*np.linalg.norm(counts))), label=labels, pad=True)
  if month:
    plt.title('Delay Percentages per Month in 2015',color='#FEF4E8')
  else:
    plt.title('Delay Percentages per Weekday in 2015',color='#FEF4E8')

  plt.axis('off')
  plt.show()

def share_pie(df):
  '''
  Plots pie chart of each airline's share within the dataset

  :param df: dataframe
  :type df: pd.DataFrame

  Returns:
  Plots pie chart of airline representation
  '''
  assert isinstance(df, pd.DataFrame)
  cmap = ['#fa8072', '#87cefa', '#f5a700', '#eb471a', '#086569', '#936400','#ac9e81','#d60800', '#E9D6EC', '#B8336A','#63A375','#e4bb95','#d5c4a1','#c8524e']
  delay2 = df.groupby(['AIRLINE']).count()
  delay2 = delay2[['DELAY']]
  delay2.columns=['Market Share']
  delay2['Market Share'].sort_index()
  delay2 = delay2.sort_values(by = 'Market Share')

  df2 = delay2[4:].copy()
  df3 = delay2['Market Share'][:4]
  new_row = pd.DataFrame(data = {

      'Market Share' : [delay2['Market Share'][:4].sum()]
  })
  new_row = new_row.rename(index={new_row.index[-1]: 'Other'})

  df2 = pd.concat([df2, new_row])

  plot = df2.plot.pie(y='Market Share',autopct='%1.2f%%', pctdistance=1.4, labels=['','','','','','','','','','',''],colors=cmap,textprops={'color':'#FEF4E8'},wedgeprops={'linewidth': 1, 'antialiased': True})
  plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, labels=['Alaska Airlines Inc.','American Eagle Airlines Inc.','JetBlue Airways','US Airways Inc.','United Air Lines Inc.','Atlantic Southeast Airlines','American Airlines Inc.','Skywest Airlines Inc.','Delta Air Lines Inc.','Southwest Airlines Co.','Other'])

  plt.title(' ',color='#FEF4E8')

  plt.xlabel('', )
  plt.ylabel('', )

  plt.show()

xlabel = 'Delay (in minutes)'
ylabel = 'Rate'
title = 'Departure Delay Distribution'
plot_hist(df.DEPARTURE_DELAY, xlabel, ylabel, title)

xlabel = 'Delay (in minutes)'
ylabel = 'Rate'
title = 'Arrival Delay Distribution'
plot_hist(df.ARRIVAL_DELAY, xlabel, ylabel, title)

xlabel = 'Airline'
ylabel = 'Percentage (%)'
title = 'Distribution of Delays by Airline'
plot_del_by(df.DELAY, 'AIRLINE', xlabel, ylabel, title)

airport = df[df.groupby('ORIGIN_AIRPORT')['ORIGIN_AIRPORT'].transform('size') > 40000]
xlabel = 'Airport'
ylabel = 'Percentage (%)'
title = 'Distribution of Delays by Top 10 Airports'
plot_del_by(df.DELAY, 'ORIGIN_AIRPORT', xlabel, ylabel, title, airport, top=10)

title = 'Density of Flight Delays for U.S. States'
plot_by_state(title, df, 'airports.csv')

xlabel = 'Date'
ylabel = 'Delays'
title = 'Distribution of Delays in 2015'
plot_by_date(xlabel, ylabel, title, df)

plot_calmap(df)

plot_square(df)

plot_square(df, month=False)

share_pie(df)
