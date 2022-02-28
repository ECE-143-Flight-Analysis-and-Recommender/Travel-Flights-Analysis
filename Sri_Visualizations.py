import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as graph

from datetime import datetime
from matplotlib.ticker import PercentFormatter

def csv_to_df(fname, max=1000000):
  assert isinstance(fname, str)
  assert isinstance(max, int)

  try:
    df = pd.read_csv(fname, nrows=max+max//2)
  except Exception as ex:
    print(f'arg max too large!')
  df = df.iloc[:, 0:24]
  df.dropna(inplace=True)
  df = df.iloc[0:max-1, :]

  return df

def plot_hist(col, xlabel, ylabel, title):
  assert isinstance(col, object) and all(col.apply(float.is_integer))
  assert all([isinstance(x, str) for x in [xlabel, ylabel, title]])

  mean = col.mean()
  std = col.std()
  total = col.value_counts().to_dict()
  min_thresh = int(mean - 2*std)
  max_thresh = int(mean + 2*std)
  plt.hist(col, density=True, bins=list(range(min_thresh, max_thresh)))
  plt.axvline(mean, color='black')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()



df = pd.read_csv('flights.csv')
df = df[df.ORIGIN_AIRPORT.apply(lambda val: not isinstance(val, int))]
df = df[df.DESTINATION_AIRPORT.apply(lambda val: not isinstance(val, int))]
df = df.iloc[:, 0:24]
df.dropna(inplace=True)
df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df['DELAY'] = np.select([df['DEPARTURE_DELAY'] > 0, df['ARRIVAL_DELAY'] > 0], [1, 1], default=0)
df['ARR_DELAY'] = np.where(df['ARRIVAL_DELAY'] > 0, 1, 0)
df['DEP_DELAY'] = np.where(df['DEPARTURE_DELAY'] > 0, 1, 0)
# print(df)

def plot_del_by(del_col, df=df, col='AIRLINE', xlabel='a', ylabel='a', title='a', top=20, figsize=(10,8)):
  delays = df.loc[del_col == 1]
  unnorm_del = list(delays[col].value_counts().sort_index())[:top]
  no_delays = df.loc[del_col == 0]
  unnorm_no_del = list(no_delays[col].value_counts().sort_index())[:top]

  count_dict = dict(df[col].value_counts().sort_index())
  keys = list(count_dict.keys())[:top]
  vals = list(count_dict.values())[:top]

  del_norm = [i/j for i,j in zip(unnorm_del, vals)]
  no_del_norm = [i/j for i,j in zip(unnorm_no_del, vals)]

  counts = [del_norm, no_del_norm]
  labels = ['DELAY', 'NO DELAY']
  xax = np.arange(len(keys))
  n = len(counts)
  fig, ax = plt.subplots(figsize=figsize)
  width = 1.5

  bot = [0]*len(keys)
  for i in range(n):
    curr_c = counts[i]
    curr_l = labels[i]
    plt.bar(xax, curr_c, width/n, label=curr_l, bottom=bot)
    bot = [x+y for x,y in zip(bot, curr_c)]

  plt.xticks(xax, keys)
  plt.legend()
  plt.show()

xlabel = 'Delay (in minutes)'
ylabel = 'Occurrance (%))'
title = 'Departure Delay Distribution'
plot_hist(df.DEPARTURE_DELAY, xlabel, ylabel, title)

xlabel = 'Delay (in minutes)'
ylabel = 'Occurrance (%))'
title = 'Arrival Delay Distribution'
plot_hist(df.ARRIVAL_DELAY, xlabel, ylabel, title)

plot_del_by(df.DELAY, col='AIRLINE')

airport = df[df.groupby('ORIGIN_AIRPORT')['ORIGIN_AIRPORT'].transform('size') > 40000]
plot_del_by(df.DELAY, df=airport, col='ORIGIN_AIRPORT', top=20)

airport_df = pd.read_csv('airports.csv')
state_dict = airport_df.set_index('IATA_CODE').to_dict()['STATE']
df['ORIGIN_STATE'] = df['ORIGIN_AIRPORT'].map(state_dict)

delay = df.loc[df.DELAY==1]
counts_dict = {'State': delay.ORIGIN_STATE.value_counts().index.to_list(),
               'Density': delay.ORIGIN_STATE.value_counts().to_list()
              }

state_to_counts = pd.DataFrame.from_dict(counts_dict)
total = state_to_counts['Density'].sum()
state_to_counts['Density'] = state_to_counts['Density'].apply(lambda x: (100*x/total))

map_plot = graph.Figure(
  data=graph.Choropleth(
    locations=state_to_counts['State'],
    z = state_to_counts['Density'],
    locationmode = 'USA-states',
    colorscale = 'Sunsetdark',
    marker_line_color='white',
    colorbar_title = "Delay density (%)"))
map_plot.update_layout(title_text = 'Density of Flight Delays for U.S. States', geo_scope='usa')
map_plot.show()

counts_dict = {'Datetime': delay.DATETIME.value_counts().index.to_list(),
               'Num': delay.DATETIME.value_counts().to_list()
              }
dates_to_counts = pd.DataFrame.from_dict(counts_dict)
dates_to_counts = dates_to_counts.sort_values(by=['Datetime'])
dates_to_counts['Sliding'] = dates_to_counts['Num'].rolling(14, min_periods=1).mean()

total = dates_to_counts['Sliding'].sum()
dates_to_counts['Sliding'] = dates_to_counts['Sliding'].apply(lambda x: 100*x/total)

plt.figure(figsize=(18,5))
markers = [datetime(2015, 1, 1), datetime(2015, 5, 23), datetime(2015, 7, 3), datetime(2015, 9, 7), datetime(2015, 11, 26), datetime(2015, 12, 25)]
sns.lineplot(x='Datetime', y='Sliding', data=dates_to_counts, label='Number of flight delays')
plt.axvline(datetime(2015, 1, 1), color='black', linewidth=2.0)
plt.axvline(datetime(2015, 5, 23), color='black', linewidth=2.0)
plt.axvline(datetime(2015, 7, 4), color='black', linewidth=2.0)
plt.axvline(datetime(2015, 9, 7), color='black', linewidth=2.0)
plt.axvline(datetime(2015, 11, 26), color='black', linewidth=2.0)
plt.axvline(datetime(2015, 12, 25), color='black', linewidth=2.0)

plt.axvspan(datetime(2015, 3, 1), datetime(2015, 3, 21), color='green', alpha=0.2, lw=0)
plt.axvspan(datetime(2015, 6, 22), datetime(2015, 8, 31), color='red', alpha=0.2, lw=0)
plt.axvspan(datetime(2015, 12, 15), datetime(2015, 12, 31), color='blue', alpha=0.2, lw=0)
plt.axvspan(datetime(2015, 1, 1), datetime(2015, 1, 4), color='blue', alpha=0.2, lw=0)

plt.show()

weekday_counts = delay.DAY_OF_WEEK.value_counts().sort_index().to_list()
total = sum(weekday_counts)
weekday_counts = [100*x/total for x in weekday_counts]
plt.bar(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], weekday_counts)
plt.xlabel('Weekday')
plt.ylabel('Delays (%)')
plt.title('Delays per Weekday')
plt.show()

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

#find some better colors
#colors = ['#88CCEE','#CC6677','#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#661100','#6699CC','#888888','#D3B484','#B3B3B3']

plot = df2.plot.pie(y='Market Share',autopct='%1.2f%%', pctdistance=1.2, labeldistance=.8)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title('Market Share')
plt.xlabel('')
plt.ylabel('')

plt.show()
