
# coding: utf-8

# ## TDI Capstone Project Proposal

# # <span style='color:skyblue;'> Unravelling the reason for suicide rate and cities fabric </span>

# Read data

# In[1]:


import pandas as pd
import numpy as np


# In[32]:


#small_dt = pd.read_csv ('temp_datalab_records_job_listings.csv', nrows=100000)


# In[2]:


df = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', dtype={"BOROUGH": object, "ZIP CODE": object})


# In[3]:


df.dtypes


# In[61]:


df.shape


# In[4]:


dt = df.loc[:,'DATE']
df.loc[:,'DATE'] = pd.to_datetime(dt, infer_datetime_format=True,
                                          errors='coerce')


# In[5]:


import datetime 

# date in yyyy/mm/dd format 
d1 = datetime.datetime(2019, 1, 1) 
df = df[df.DATE < d1]
max(df.DATE)


# ** Total number of persons injured **

# In[6]:


np.nansum(df['NUMBER OF PERSONS INJURED'])


# ** Proportion of collision in Brooklyn in 2016 **

# In[87]:


print(set(df['BOROUGH']))


# In[104]:


filtr_brook_16 = (((df['DATE'] < datetime.datetime(2017, 1,1)) &
          (df['DATE'] > datetime.datetime(2015, 12, 31))) &
         (df['BOROUGH']=='BROOKLYN'))

sum(filtr_brook_16)/sum( df['BOROUGH'].notnull())


# ** Proportion of cyclists injured or killed in 2016 **

# In[17]:


filtr_2016 =((df['DATE'] < datetime.datetime(2017, 1,1)) &
          (df['DATE'] > datetime.datetime(2015, 12, 31))) 
accidents_2016 = df[filtr_2016]
total_cyclist_2016 = np.nansum(accidents_2016['NUMBER OF CYCLIST INJURED']) + np.nansum(accidents_2016['NUMBER OF CYCLIST KILLED'])

# ratio total cyclist injured or killed in 2016 divided by total number of accidents in 2016
total_cyclist_2016/accidents_2016.shape[0]


# ** Accident per capita involving alcohol in 2017 **

# In[191]:


filtr_2017 =((df['DATE'] < datetime.datetime(2018, 1,1)) &
          (df['DATE'] > datetime.datetime(2016, 12, 31))) 
accidents_2017 = df[filtr_2017]
alcohol_17_filter = accidents_2017.loc[:, 'CONTRIBUTING FACTOR VEHICLE 1':'CONTRIBUTING FACTOR VEHICLE 5'].apply(lambda x: x.str.contains('Alcohol').notnull())
alcohol_17_filter = alcohol_17_filter.any(axis=1)

borough_2017_alcohol = accidents_2017[alcohol_17_filter] #['BOROUGH']
borough_2017_alcohol = borough_2017_alcohol[borough_2017_alcohol['BOROUGH'].notnull()]
borough_alcohol_2017_total = borough_2017_alcohol.groupby('BOROUGH')['BOROUGH'].count()


# In[193]:


borough_alcohol_2017_total


# In[192]:


# Since the data is small I am going to copy and paste rather than scraping it

borough_pop =pd.Series({'BRONX': 1471160,'BROOKLYN': 2648771 , 'MANHATTAN': 1664727 , 'QUEENS': 2358582, 'STATEN ISLAND': 479458})
(borough_alcohol_2017_total/borough_pop).max()


# ** Max vehicles per zipcode in 2016 **

# In[214]:


#df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)

vehicles_16_filter = accidents_2016.loc[:, 'VEHICLE TYPE CODE 1':'VEHICLE TYPE CODE 5'] #.apply(lambda x: x.notnull())
vehicles_2016 = vehicles_16_filter.count(axis=1) #.sum(axis=1)
accidents_2016 = accidents_2016.assign(total_vehicles=vehicles_2016)

zipcode_2016_total = accidents_2016.groupby('ZIP CODE')['total_vehicles'].count()
zipcode_2016_total.max()


# ** Yearly trend **

# In[19]:


filtr_13_18 =((df['DATE'] < datetime.datetime(2019, 1,1)) &
          (df['DATE'] > datetime.datetime(2012, 12, 31))) 
accidents_13_18 = df[filtr_13_18] 
accidents_13_18 = accidents_13_18.assign(year = accidents_13_18['DATE'].dt.year)
yearly_accidents = accidents_13_18.groupby('year')['year'].count()
yearly_accidents


# In[38]:


x= np.array(yearly_accidents.index).reshape(-1,1)
y = yearly_accidents.values.reshape(-1,1)
y.shape


# In[40]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, y)
reg.coef_


# Winter collisions

# In[44]:


filtr_2017 =((df['DATE'] < datetime.datetime(2018, 1,1)) &
          (df['DATE'] > datetime.datetime(2016, 12, 31))) 
accidents_2017 = df[filtr_2017]

vehicles_17_filter = accidents_2017.loc[:, 'VEHICLE TYPE CODE 1':'VEHICLE TYPE CODE 5'] #.apply(lambda x: x.notnull())
vehicles_2017 = vehicles_17_filter.count(axis=1) #.sum(axis=1)
accidents_2017 = accidents_2017.assign(total_vehicles=vehicles_2017)
accidents_2017 = accidents_2017.assign(month = accidents_2017['DATE'].dt.month)

monthly_accidents_2017= accidents_2017.groupby('month')['total_vehicles'].apply(lambda x: sum(x>2)/x.count())
monthly_accidents_2017


# In[65]:


from scipy.stats import chisquare
chisquare(monthly_accidents_2017.loc[[1,5]])


# <h3> Collisions per square kilometers </h3>

# In[77]:


accidents_2017_zipcode = accidents_2017.groupby('ZIP CODE').filter(lambda x: x['UNIQUE KEY'].count() > 1000)

# Manually got the lat lon boundary for NY city from Google map
NY_lat_boundary = [40.496058, 40.915466]
NY_lon_boundary = [-74.255220,-73.701629]

# remove row with lat-lon outside this boundary
lat_filter = (accidents_2017_zipcode['LATITUDE'] < NY_lat_boundary[1]) & (accidents_2017_zipcode['LATITUDE'] > NY_lat_boundary[0])
lon_filter = (accidents_2017_zipcode['LONGITUDE'] < NY_lon_boundary[1]) & (accidents_2017_zipcode['LONGITUDE'] > NY_lon_boundary[0])
lat_lon_filter = lat_filter & lon_filter
accidents_2017_ny = accidents_2017_zipcode[lat_lon_filter]


# In[92]:


accidents_2017_ny_zipcode = accidents_2017_ny.groupby('ZIP CODE')['LATITUDE','LONGITUDE'].agg([np.std, np.mean])
accidents_2017_ny_zipcode['LATITUDE']['mean']


# ### Compute elipse radius and area 

# The formula used is **equirectangular approximation**.
# 
# Ref: https://www.movable-type.co.uk/scripts/latlong.html

# In[97]:


import math
earth_radius = 6371  # km
lon_radius = earth_radius * accidents_2017_ny_zipcode['LONGITUDE']['std'] # * math.cos(accidents_2017_ny_zipcode['LATITUDE']['mean'])
lat_radius = earth_radius * accidents_2017_ny_zipcode['LATITUDE']['std']

# Zip code area using ellipse area formula

zipcode_area = math.pi * lon_radius * lat_radius

# Collisions per square kilometer

accidents_2017_zipcode_km = accidents_2017_ny.groupby('ZIP CODE')['UNIQUE KEY'].count()/zipcode_area
max(accidents_2017_zipcode_km)

