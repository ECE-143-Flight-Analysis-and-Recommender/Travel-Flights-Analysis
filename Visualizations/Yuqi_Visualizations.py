import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import plotly

plt.style.use('ggplot')

df = pd.read_csv("flights.csv")

df_dow = pd.DataFrame(df.groupby(['MONTH','DAY','DAY_OF_WEEK']).size(), columns=['counts'])
df_dow.reset_index(inplace = True)

d = {1:'Mon',2:'Tues',3:'Wed',4:'Thurs',5:'Fri',6:'Sat',7:'Sun'}
df_dow['day_of_week'] = df_dow['DAY_OF_WEEK'].replace(d)

ax = df_dow.pivot_table('counts', index='MONTH', columns='day_of_week',aggfunc=np.mean,sort=True).plot(figsize=(16,9))
ax.legend(['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
ax.set_ylabel('average count')

df_airline = pd.read_csv("airlines.csv")

df['AIRLINE'] = df['AIRLINE'].map(df_airline.set_index('IATA_CODE')['AIRLINE'])

df['departute_hour'] = np.floor((df['SCHEDULED_DEPARTURE']-1)/100).astype(np.int8)
df['arrival_hour'] = np.floor((df['SCHEDULED_ARRIVAL']-1)/100).astype(np.int8)

df['status'] = 0
df.loc[(df['ARRIVAL_DELAY']>0) & (df['DEPARTURE_DELAY']>0),'status'] = 1 #delay
df.loc[df['CANCELLATION_REASON'].notnull(),'status'] = 2 #cancel

#nan in 0-5
df_rate_arr = pd.DataFrame(df[df['status']==1].groupby(['arrival_hour','AIRLINE']).size()/df.groupby(['arrival_hour','AIRLINE']).size(),columns=['delay_rate'])
df_rate_arr = df_rate_arr.pivot_table('delay_rate', index='AIRLINE', columns='arrival_hour')
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(df_rate_arr[list(range(6,24))], cmap='RdBu_r', ax=ax, linecolor='black', linewidth=0.01,cbar_kws={'label': 'delay_rate'})


#nan in 0-5
df_rate_dep = pd.DataFrame(df[df['status']==1].groupby(['departute_hour','AIRLINE']).size()/df.groupby(['departute_hour','AIRLINE']).size(),columns=['delay_rate'])
df_rate_dep = df_rate_dep.pivot_table('delay_rate', index='AIRLINE', columns='departute_hour')
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(df_rate_dep[list(range(6,24))], cmap='RdBu_r', ax=ax, linecolor='black', linewidth=0.01,cbar_kws={'label': 'delay_rate'})

df['speed'] = df['DISTANCE']/df['AIR_TIME']*60
df1 = df[df['speed'].notnull()]
df2 = pd.DataFrame(df1.groupby(['AIRLINE']).size(),columns=['count'])
df2['avg_speed_not_delay'] = df1[df1['status']==0].groupby(['AIRLINE'])['speed'].mean()
df2['avg_speed_delay'] = df1[df1['status']==1].groupby(['AIRLINE'])['speed'].mean()
df2['arr_delay_rate'] = df1[df1['status']==1].groupby(['AIRLINE']).size()/df1.groupby(['AIRLINE']).size()
df2['arr_delay'] = df1.groupby(['AIRLINE'])['ARRIVAL_DELAY'].mean()
df2['dep_delay'] = df1.groupby(['AIRLINE'])['DEPARTURE_DELAY'].mean()

sns.relplot(x="avg_speed_not_delay", y="avg_speed_delay", hue="AIRLINE", size="count",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df2)

sns.relplot(x="dep_delay", y="arr_delay", hue="AIRLINE", size="count",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df2)

df_delay_type = pd.Series([len(df[df['AIR_SYSTEM_DELAY']>0]),len(df[df['SECURITY_DELAY']>0]),
                              len(df[df['AIRLINE_DELAY']>0]),len(df[df['LATE_AIRCRAFT_DELAY']>0]),
                              len(df[df['WEATHER_DELAY']>0])],
                             ['air_system','security','airline','late_aircraft','weather'])

colors = sns.color_palette('pastel')
plt.pie(df_delay_type,labels=df_delay_type.index,autopct='%1.1f%%',colors=colors)
plt.title("reasons for delay")
plt.show()

sns.barplot(x=df_delay_type.index, y=df_delay_type.values)

df_airport = pd.read_csv("airports.csv")
df['route'] = tuple(zip(df['ORIGIN_AIRPORT'],df['DESTINATION_AIRPORT']))
df['route'] = df['route'].apply(sorted)
df['route'] = df['route'].apply(tuple)

df['latitude_origin'] = df['ORIGIN_AIRPORT'].map(df_airport.set_index('IATA_CODE')['LATITUDE'])
df['latitude_des'] = df['DESTINATION_AIRPORT'].map(df_airport.set_index('IATA_CODE')['LATITUDE'])
df = df[df['latitude_origin'].notnull() & df['latitude_des'].notnull()].reset_index(drop=True)

df_route = pd.DataFrame(df.groupby(['route']).size(), columns=['count'])
df_route = df_route[df_route['count']>5000].reset_index()
df_route['origin'],df_route['destination'] = zip(*df_route['route'])
df_route['latitude_origin'] = df_route['origin'].map(df_airport.set_index('IATA_CODE')['LATITUDE'])
df_route['longtitude_origin'] = df_route['origin'].map(df_airport.set_index('IATA_CODE')['LONGITUDE'])
df_route['latitude_des'] = df_route['destination'].map(df_airport.set_index('IATA_CODE')['LATITUDE'])
df_route['longtitude_des'] = df_route['destination'].map(df_airport.set_index('IATA_CODE')['LONGITUDE'])

df_route['count_delay'] = df_route['route'].map(df[df['status']==1].groupby(['route']).size())
df_route.loc[df_route['count_delay'].isnull(),'count_delay'] = 0

df_route['delay_rate'] = df_route['count_delay']/df_route['count']

df_airport['delay_rate'] = df_airport['IATA_CODE'].map(df[df['status']==1].groupby('ORIGIN_AIRPORT').size()/df.groupby('ORIGIN_AIRPORT').size())
df_airport['count'] = df_airport['IATA_CODE'].map(df.groupby('ORIGIN_AIRPORT').size())
df_airport = df_airport[df_airport['count'].notnull()]

fig = px.scatter_mapbox(df_airport[df_airport['count']>1000], lat="LATITUDE", lon="LONGITUDE", color="delay_rate", size="count",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=3)

c_min,c_max=min(df_airport['delay_rate']),max(df_airport['delay_rate'])
fig.update_geos(projection_type="orthographic")
for i in range(len(df_route)):
    p_of_cmap=(df_route['delay_rate'][i]-c_min)/(c_max-c_min)
    fig.add_trace(
        go.Scattermapbox(
            lon = [df_route['longtitude_origin'][i], df_route['longtitude_des'][i]],
            lat = [df_route['latitude_origin'][i], df_route['latitude_des'][i]],
            mode = 'lines',
            line = dict(width = 1, color = plotly.colors.sample_colorscale(px.colors.cyclical.IceFire,p_of_cmap)[0]),
            opacity = float(df_route['count'][i]) / float(df_route['count'].max()),
            showlegend=False,
        )
    )


fig.update_layout(mapbox_style="light",
                  mapbox_accesstoken="pk.eyJ1IjoibGFsYWxhMjAyMSIsImEiOiJja3drNGl0djAxb2phMnVubzZkYWhmZ3hkIn0.TbWtnv71s5niti55m-KJ6g",
                  title='Delay rate of flights and airports in US'
)

fig.show()
