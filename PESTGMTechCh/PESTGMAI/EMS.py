#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import get_ipython




import time
from sklearn.ensemble import IsolationForest
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose


import warnings
warnings.filterwarnings("ignore")


# In[2]:


o_data = pd.read_csv('HomeC14.csv')
data = o_data.copy()


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.describe().T


# In[6]:


pd.set_option('display.max_columns', None)
data.columns


# In[7]:


data.columns = [col.replace(' [kW]', '') for col in data.columns]


# In[8]:


data.shape


# In[9]:


data['Furnace'] = data[['Furnace 1', 'Furnace 2']].sum(axis=1)
data['Kitchen'] = data[['Kitchen', 'Kitchenette', ]].sum(axis=1)
data


# In[10]:


data.drop(['Furnace 1', 'Furnace 2', 'Kitchenette'], axis=1, inplace=True)
data


# In[11]:


data['icon'].nunique()


# In[12]:


data['summary'].unique()


# In[13]:


data.isna().sum()


# In[14]:


data.isna().sum()


# In[15]:


data[data.isnull().any(axis=1)]


# In[16]:


data = data.drop(index=503910,axis=0)


# In[17]:


data.tail(5)


# **Column 'cloudCover' has a datatype of 'Object' instead of 'float'. This is due to it having string values 'cloudCover' in them. This may be due to some error during data collection.**

# In[18]:


data.cloudCover.dtype


# In[19]:


data['cloudCover'].unique()


# In[20]:


data[data.cloudCover == 'cloudCover'].shape


# There are 58 invalid entries of ‘cloudCover’. So replacing this to valid values and changing the datatype of column ‘cloudCover’ to float.

# In[21]:


data['cloudCover'].replace('cloudCover', method='bfill', inplace=True)


# In[22]:


data['cloudCover'] = data['cloudCover'].astype('float')


# In[23]:


data.cloudCover.dtype


# ### **The 'time' feature is in 'unixtimestamp' and of datatype 'Object' and need to be converted to type 'Datetime'**

# In[24]:


data['time'].dtypes


# In[25]:


pd.to_datetime(data['time'], unit='s')


# ## Setting date column to date time format in unit as seconds

# In[26]:


data['time'] = pd.to_datetime(data['time'], unit='s')


# In[27]:


data


# ### **Changing the feature 'time' into increments of 1 minute.**
# 

# In[28]:


data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min'))

data_out=data.copy()


# In[29]:


data['time'].dtypes


# In[30]:


data


# In[31]:


data['year'] = data['time'].apply(lambda x:x.year)
data['month'] = data['time'].apply(lambda x:x.month)
data['day'] = data['time'].apply(lambda x:x.day)
data['hour'] = data['time'].apply(lambda x:x.hour)
data['minute'] = data['time'].apply(lambda x:x.minute)


# In[32]:


def hours_to_timing(hour):
    if hour in [22, 23, 0, 1, 2, 3]:
        timing = 'Night'
    elif hour in range(4, 12):
        timing = 'Morning'
    elif hour in range(12, 17):
        timing = 'Afternoon'
    elif hour in range(17, 22):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing


# In[33]:


data['timing'] = data['hour'].apply(hours_to_timing)


# In[34]:


data


# In[35]:


data[['year', 'month',
       'day', 'hour', 'minute', 'timing']].dtypes


# In[36]:


def Season(month):
    if month in [1,2,3]:
        season = 'Winter'
    elif month in [4,5,6]:
        season = 'Spring'
    elif month in [7,8,9]:
        season = 'Summer'
    elif month in [10,11,12]:
        season = 'Autumn'            
    else:
        season='X'
            
    return season


# In[37]:


data['season'] = data['month'].apply(Season)


# In[38]:


data


# ## **Identifying Duplicates**

# **Majority of the values are same in 'gen' and 'solar' and rest are null values in gen. So we are dropping gen**
# 
# Same applies of column 'use' also.So dropping it as well

# In[39]:


q=np.where(data['use'] == data['House overall'])


# In[40]:


r=np.where(data['gen'] == data['Solar'])


# In[41]:


print(q)
print(r)


# In[42]:


data = data.drop(['gen','use'],axis=1)


# In[43]:


data.info()


# ## **Grouping feature into categorical data and numerical data**

# In[44]:


# Grouping feature into categorical data and numerical data
categorical_features=[features for features in data.columns if data[features].dtypes=="O" and features not in ['timing']]
categorical_features


# In[45]:


numerical=[features for features in data.columns if data[features].dtypes!="O" and features not in ['time','year', 'month','day', 'hour', 'minute']]
numerical


# In[46]:


len(numerical)


# In[47]:


object_nan = [feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes=='O']
object_nan


# In[48]:


numerical_nan = [feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes!='O']
numerical_nan


# ## Grouping feature into energy data and weather data
# 

# In[49]:


energy_data = data[['House overall', 'Dishwasher', 'Home office',
       'Fridge',  'Garage door',  'Well', 'Microwave',
       'Living room', 'Furnace', 'Kitchen']]
weather_data = data[['Solar','temperature', 'humidity', 'visibility', 'apparentTemperature',
       'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity',
       'dewPoint', 'precipProbability']]


# In[50]:


# Converting index from 'int' to 'datetime'

energy_data.index = pd.to_datetime(energy_data.index)
weather_data.index = pd.to_datetime(weather_data.index)


# In[51]:


# Setting energy data index to 'datetime'
energy_data = energy_data.set_index(data['time'])

# Setting weather data index to 'datetime'

weather_data = weather_data.set_index(data['time'])


# In[52]:


energy_per_day = energy_data.resample('D').sum()
energy_per_week = energy_data.resample('W').sum()


# In[53]:


energy_per_month = energy_data.resample('M').sum()
energy_per_hour = energy_data.resample('H').sum()


# In[54]:


energy_data.index.dtype
weather_data.index.dtype


# In[55]:


energy_data


# In[56]:


weather_per_day = weather_data.resample('D').mean()
weather_per_month = weather_data.resample('M').mean()
weather_per_week = weather_data.resample('W').mean()

weather_per_hour = weather_data.resample('H').mean()


# In[57]:


weather_per_hour 


# In[58]:


weather_per_hour


# 
# ## Time series analyis

# In[59]:


# Time series analysis for energy per day
import statsmodels.api
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

datause=energy_per_day.iloc[:,0].values
#fig,ax=plt.subplots(figsize=(15,10))
plt.rcParams['figure.figsize'] = (14, 9)
result = seasonal_decompose(energy_per_day['House overall'],model='additive')
result.plot()
plt.show()


# In[60]:


# Time series analysis for energy per Hour
import statsmodels.api
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

datause=energy_per_hour.iloc[:,0].values
#fig,ax=plt.subplots(figsize=(15,10))
plt.rcParams['figure.figsize'] = (14, 9)
result = seasonal_decompose(energy_per_hour['House overall'],model='additive')
result.plot()
plt.show()


# In[61]:


# Time series analysis for energy per day

weather=weather_per_day.iloc[:,0].values
#fig,ax=plt.subplots(figsize=(15,10))
plt.rcParams['figure.figsize'] = (14, 9)
seasonal_decompose(weather_per_day[['Solar']]).plot()
result = adfuller(weather)
plt.show()


# Solar Generation is maximum in the months of April-May

# # **Data Visualization**

# In[62]:


# Plotting total energy consumption per day

plt.figure(figsize=(15,7))
plt.title("Per Day - Overall Energy Consumption")
sns.lineplot(data = energy_per_day['House overall'], dashes=False)
plt.show()


# nference- We can see an increasing trend in energy consumption per day from August and it reaches the peak in September and again starts falling by the end of September 

# In[63]:


# Plotting total energy consumption per month

plt.figure(figsize=(15,7))
plt.title("Overall Energy Consumption Per Month")
sns.lineplot(data = energy_per_month['House overall'], dashes=False)
plt.show()


# **Inference-The highest energy consumption in the year falls in the months of August and September followed by the month of February.**
# **Whereas the lowest consumption is in the month of January. **

# In[64]:


# Plotting the Distribution of energy per hour

plt.figure(figsize=(8,6))
sns.distplot(energy_per_hour['House overall'])
plt.title('Total Energy Consumption Distribution')
plt.show()


# In[65]:


# Plotting the Distribution of Solar generation per hour
plt.figure(figsize=(8,6))
sns.distplot(weather_per_hour['Solar'])
plt.title('Total Energy Generation Distribution')
plt.show()



# In[66]:


plt.plot(weather_per_hour["windSpeed"])


# In[67]:


pp_data = energy_data[['Dishwasher', 'Fridge', 'Garage door', 'Well', 'Microwave', 'Furnace']]
room_data = energy_data[['Home office', 'Living room', 'Kitchen']]


# In[68]:


# Plotting per month energy consumption of each device

plt.figure(figsize=(15,7))
plt.title("Per Month Energy Consumption of Each Appliance")
sns.lineplot(data = energy_per_month[pp_data.columns], dashes=False)
plt.show()


# In[69]:


plt.figure(figsize=(15,7))
plt.title("Room energy consumption per month")
sns.lineplot(data = energy_per_month[room_data.columns], dashes=False)
plt.show()


# **From above it can be see that 'Home office' has the highest consumption in the among rooms and the 'Kitchen' has the lowest consumption.**
# 

# In[70]:


plt.figure(figsize=(15,6))
plt.title("Solar Energy Generation per month")
sns.lineplot(data = weather_per_month['Solar'],dashes=False)
plt.show()


# In[71]:


# Plotting Frequency graph showing distribution
freqgraph = energy_per_hour.select_dtypes(include=["float",'int'])
freqgraph.hist(figsize=(20,15))
plt.show()


# In[72]:


# Plotting total Energy Consumption with respect to time in a day

plt.figure(figsize=(12,6))
sns.barplot(x=data['hour'], y=data['House overall'])
plt.title('Overall Energy Cosumption at each hour of the day')
plt.show()


# In[73]:


plt.figure(figsize=(12,6))
sns.barplot(x=data['hour'], y=data['Solar'])
plt.title('Overall Solar Generation at each hour of the day')
plt.show()


# In[74]:


for i in energy_data:
  plt.figure(figsize=(12,6))
  sns.barplot(x=data['timing'], y=data[i])
  plt.title('Power Consumption of {} at different timings of the day'.format(i))
  plt.show()


# In[75]:


plt.figure(figsize=(12,6))
sns.barplot(x=data['timing'], y=data['Solar'])
plt.title('Solar generation at different timings of the day')
plt.show()


# In[76]:


#Plot of Average Temperature per month
plt.figure(figsize=(15,5))
plt.ylabel('°F')
plt.title("Mean Temperature Per Month")
sns.lineplot(data = weather_per_month[['temperature', 'apparentTemperature']],dashes=False)


# In[77]:


plt.figure(figsize=(15,5))
plt.title("Mean Humidity Per Month")
sns.lineplot(data = weather_per_month['humidity'],dashes=False)


# ## **Calulating Total Energy Consumption per month for rooms and appliances respectively**

# In[78]:


rooms_energy_per_month = energy_per_month[['Home office', 'Living room', 'Kitchen']]
app_energy_per_month = energy_per_month[['Dishwasher', 'Fridge', 'Garage door', 'Well', 'Microwave', 'Furnace']]


# In[79]:


rooms_tot_consum_per_month = rooms_energy_per_month.sum()
app_tot_consum_per_month = app_energy_per_month.sum()


# In[80]:


print('The total energy consumption of rooms per month is :\n', rooms_tot_consum_per_month, '\n')
print('The total energy consumption of appliances per month is :\n', app_tot_consum_per_month)


# In[81]:


# Plotting Pie Chart to show % Consumption of rooms per month  

rooms_tot_consum_per_month.plot(kind="pie", autopct='%.2f', figsize=(8,8))
plt.title("Percentage Consumption for rooms per month")
plt.ylabel('%')
plt.show()


# In[82]:


# Plotting Pie Chart to show % Consumption of devices per month 

app_tot_consum_per_month.plot(kind="pie", autopct='%.2f', figsize=(8,8))
plt.title("Percentage Consumption for Appliances per month")
plt.ylabel('%')
plt.show()


# ## Regplot showing regression line between two parameters and the linear relationship 
# 
# how tight the datapoints occuring around them determines the extent of linear relationship
# 
# If the datapoints appears close to the regression line. the line seems to be a good fit regression line

# In[83]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=energy_per_hour['House overall'], y= weather_per_hour['temperature'])
plt.title('Total Energy Consumption vs Temperature')
plt.show()


# plt.show()
# Inference- 
# 
# When temperature is in the range of 60 to 80 degree F energy consumption seems to be the most.
# 
# When Temperatures occurs between 0 to 10 degree F House Overall it at its minimum range

# In[84]:


plt.figure(figsize=(8,6))
sns.regplot(x=weather_per_hour['Solar'], y= weather_per_hour['temperature'])
plt.title('Total Energy Generation vs Temperature')
plt.show()


# There doesnt seem to be any linear relationship between Solar generation and temperature in our data
# 

# In[85]:


plt.figure(figsize=(8,6))
sns.scatterplot(x = energy_per_day['Kitchen'], y = weather_per_day['temperature'])
plt.title('Kitchen vs Temperature')
plt.show()


# In[86]:


plt.figure(figsize=(8,6))
sns.scatterplot(x = energy_per_day['Furnace'], y = weather_per_day['temperature'])
plt.title('Furnace vs Temperature')
plt.show()


# In[87]:


plt.figure(figsize=(8,6))
sns.regplot(x = energy_per_day['Fridge'], y = weather_per_day['dewPoint'])
plt.title('Fridge vs dewpoint')
plt.show()


# In[88]:


plt.figure(figsize=(8,6))
sns.regplot(x = energy_per_day['Fridge'], y = weather_per_day['temperature'])
plt.title('Fridge vs Temperature')
plt.show()


# ### **Usage of different appliabnces are ploted using univcarite plots below**

# In[89]:


plt.figure(figsize=(10,10))

plt.subplot(3,1,1)
plt.plot(energy_per_hour['Kitchen'],np.zeros_like(energy_per_hour['Kitchen']),'o',color='green')
plt.xlabel('Proportion of Kitchen usage of Overall Power')
plt.subplot(3,1,2)
plt.plot(energy_per_hour['Fridge'],np.zeros_like(energy_per_hour['Fridge']),'o',color='Orange')
plt.xlabel('Proportion of fridge usage of Overall Power')
plt.subplot(3,1,3)
plt.plot(energy_per_hour['Furnace'],np.zeros_like(energy_per_hour['Furnace']),'o',color='Purple')   
plt.xlabel('Proportion of furnace usage of Overall Power')
plt.show()


# In[90]:


plt.figure(figsize=(10,10))
plt.subplot(4,1,1)
plt.plot(energy_per_hour['Home office'],np.zeros_like(energy_per_hour['Home office']),'o',color='Purple')
plt.xlabel('Proportion of Home Office usage of Overall Power')
plt.subplot(4,1,2)
plt.plot(energy_per_hour['Dishwasher'],np.zeros_like(energy_per_hour['Dishwasher']),'o',color='Yellow')
plt.xlabel('Proportion of Dishwasher usage of Overall Power')
plt.subplot(4,1,3)
plt.plot(energy_per_hour['Garage door'],np.zeros_like(energy_per_hour['Garage door']),'o',color='green')
plt.xlabel('Proportion of Garage doorusage of Overall Power')
plt.show()


# # Bivariate Analysis

# In[91]:


sns.FacetGrid(weather_per_hour).map(plt.scatter,'temperature','humidity').add_legend()
plt.show()


# ### Facet Grid of temperature and dewpoint= showing positive correlation

# In[92]:


sns.FacetGrid(weather_per_hour).map(plt.scatter,'dewPoint','temperature').add_legend()
plt.show()


# **Temperature and dewpoint has a positive correlation**

# In[93]:


sns.FacetGrid(weather_per_hour).map(plt.scatter,'dewPoint','pressure').add_legend()
plt.show()


# ### **Pair Plot of weather parameters with solar energy generation and Power Consumption**

# In[94]:


weather_energy =pd.concat([weather_data[['temperature', 'humidity', 'pressure', 'windSpeed', 'cloudCover','Solar']], energy_data[['House overall']]], axis=1)
weather_energy


# In[95]:


sns.pairplot(energy_per_hour)


# ## **Distribution of Numerical Data**

# Distplot

# In[96]:


numerical


# In[97]:


for feature in energy_per_hour:
  sns.distplot(energy_per_hour[feature], kde=True)
  plt.title('Hericalistogram for {}'.format(feature))
  plt.show()


# In[98]:


weather_per_hour.skew()


# Null Value Handlingµ

# In[99]:


data.isna().sum()


# In[100]:


data['icon'].fillna(data['icon'].mode()[0],inplace=True)
data['apparentTemperature'].fillna(data['apparentTemperature'].median(),inplace=True)
data.isna().sum()


# Label encoding categorical column

# In[101]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()#le has operations for label encoding


data['summary']=le.fit_transform(data['summary'])


# ## **Again taking energy per hour and weather per hour after filling null values, then resampling and concating into data per hour**

# In[102]:


energy_data1 = data[['House overall', 'Dishwasher', 'Home office',
       'Fridge',  'Garage door',  'Well', 'Microwave',
       'Living room','Solar', 'Furnace','Kitchen']]
weather_data1 = data[['temperature', 'humidity', 'visibility', 'apparentTemperature',
       'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity',
       'dewPoint', 'precipProbability']]
energy_data1.index = pd.to_datetime(energy_data.index)
weather_data1.index = pd.to_datetime(weather_data.index)


# In[103]:


energy_data1 = energy_data1.set_index(data['time'])
energy_per_hour1 = energy_data1.resample('H').sum()
weather_data1 = weather_data1.set_index(data['time'])
weather_per_hour1 = weather_data1.resample('H').mean()
data_per_hour= pd.concat([energy_per_hour1, weather_per_hour1],axis=1)
data_per_hour


# ## **Correlation Table Weather data and Energy Data**
# 

# In[104]:


consumptions = energy_per_hour1.columns.tolist()
weather = weather_per_hour1.columns.tolist()
lists = [consumptions,]
for j in weather_per_hour1:
  correlations = []
  for i in energy_per_hour1:
    cor = data[i].corr(data[j])
    correlations.append(cor)
  lists.append(correlations)

names=['consumptions',]
for i in weather:
  names.append(i+'_corr')
data_corr = pd.DataFrame(np.column_stack(lists), columns=names).set_index('consumptions')

for i in data_corr.columns[:].tolist():
  data_corr[i] = data_corr[i].apply(float)


# In[105]:


data_corr.style.applymap(lambda x: "background-color: red" if x > 0.1 else "background-color: orange" if x < -0.1 else "background-color: white")


# In[106]:


dfw = pd.concat([weather_per_hour1[['temperature', 'humidity', 'visibility', 'apparentTemperature','pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity','dewPoint','precipProbability']],energy_per_hour1[['House overall','Solar']]],axis=1)
corrmatrix = dfw.corr()
plt.subplots(figsize=(20,10))
sns.heatmap(corrmatrix,annot=True,cmap = 'YlGnBu')


# In[107]:


dfe= data_per_hour[['House overall','Dishwasher', 'Home office','Fridge',  'Garage door',  'Well', 'Microwave','Living room','Solar', 'Furnace','Kitchen']]
corrmatrix = dfe.corr()
plt.subplots(figsize=(20,10))
sns.heatmap(corrmatrix,annot=True,cmap = 'YlGnBu')


# In[108]:


data_per_hour = data_per_hour.drop(['apparentTemperature','precipProbability','visibility',
       'cloudCover', 'windBearing', 'precipIntensity'],axis=1)


# In[109]:


numerical=[features for features in data_per_hour.columns if data_per_hour[features].dtypes!="O"]


# In[110]:


for feature in numerical:
  data_per_hour.boxplot(column= feature )
  plt.xlabel(feature)
  plt.title(feature)
  plt.show()


# In[111]:


def find_outliers_IQR(data_per_hour):
    q1=data_per_hour.quantile(0.25)
    q3=data_per_hour.quantile(0.75)
    IQR=q3-q1
    outliers = data_per_hour[((data_per_hour<(q1-1.5*IQR)) | (data_per_hour>(q3+1.5*IQR)))]
    
    return outliers

for feature in numerical:
    outliers = find_outliers_IQR(data_per_hour[feature])

    print(feature)
    print('number of outliers: '+ str(len(outliers)))
    print('max outlier value: '+ str(outliers.max()))
    print('min outlier value: '+ str(outliers.min()))
    print('% of outliers: '+ str(len(outliers)/(len(data_per_hour[feature]))*100))
    print('\n')


# In[112]:


data_outlier=data_per_hour.copy()
data_outlier.reset_index(inplace=True)


# In[113]:


def Season(month):
  if month in [1,2,3]:
    season = 'Winter'
  elif month in [4,5,6]:
     season = 'Spring'

  elif month in [7,8,9]:
     season = 'Summer' 
  elif month in [10,11,12]:
    season = 'Autumn'

  else:
    season='X'

  return season  


# In[114]:


data_outlier['year'] = data_outlier['time'].apply(lambda x: x.year)
data_outlier['month'] = data_outlier['time'].apply(lambda x: x.month)
data_outlier['day'] = data_outlier['time'].apply(lambda x: x.day)

data_outlier['hour'] = data_outlier['time'].apply(lambda x: x.hour)
data_outlier['minute'] = data_outlier['time'].apply(lambda x: x.minute)
data_outlier['season'] = data_outlier['month'].apply(Season)
data_outlier


# In[115]:


for feature in numerical:
    plt.figure(figsize=(15,10))
    sns.boxplot(x = 'season', y = feature, data = data_outlier)
    plt.title(feature)
    plt.show()
    


# **Checking for Skewness**
# 

# In[116]:


numerical=[features for features in data_per_hour.columns if data_per_hour[features].dtypes!="O" ]


data_per_hour[numerical].agg(['skew', 'kurtosis']).transpose()


# ## **Using Log and Box cox to transform data and bring down skewness**

# In[117]:


numerical


# In[118]:


df1=data_per_hour.copy()
df1


# In[119]:


for i in df1[[ 'Well', 'Microwave', 'Living room','Dishwasher','Kitchen']]:
  df1[i] = df1[i].apply(lambda x:x**(1/3))
  print('original skewness: {}'.format(i),data_per_hour[i].skew(),'\nSkewness after transformation :',df1[i].skew())
  


# In[120]:


df1['windSpeed'] = df1['windSpeed'].apply(lambda x: x ** (1 / 2))
print('original skewness: {}'.format(i), data_per_hour['windSpeed'].skew(), '\nSkewness after transformation :',df1['windSpeed'].skew())
df1['Home office'] = df1['Home office'].apply(lambda x:x**(1/2))
print('original skewness: {}'.format(i),data_per_hour['Home office'].skew(),'\nSkewness after transformation :',df1['Home office'].skew())
df1['House overall'] = df1['House overall'].apply(lambda x:x**(1/3))
print('original skewness:{}'.format(i),data_per_hour['House overall'].skew(),'\nSkewness after transformation :',df1['House overall'].skew())
df1['dewPoint'] = df1['dewPoint'].apply(lambda x:x**(1/0.25))
print('original skewness:{}'.format(i),data_per_hour['dewPoint'].skew(),'\nSkewness after transformation :',df1['dewPoint'].skew())


# In[121]:


for i in df1[[ 'Fridge', 'Solar']]:
  df1[i] = df1[i].apply(lambda x:x**(1/-3))
  print('original skewness : {}'.format(i),data_per_hour[i].skew(),'\nSkewness after transformation :',df1[i].skew())


# In[122]:


df1['Garage door'] = df1['Garage door'].apply(lambda x:x**(1/3))
print('original skewness: {}'.format(i),data_per_hour['Garage door'].skew(),'\nSkewness after transformation :',df1['Garage door'].skew())


# In[123]:


df1.skew()


# In[124]:


df1


# In[125]:


(df1 < 0).sum().sum()


# In[126]:


for i in df1.columns:
  print(np.where((df1[i] < 0)))


# In[127]:


df1.skew()


# In[128]:


for col in numerical:
  print(col)
  print('Skew :', round(df1[col].skew(), 2))
  plt.figure(figsize = (15, 4))
  plt.subplot(1, 2, 1)
  df1[col].hist(grid=False)
  plt.ylabel('count')
  plt.subplot(1, 2, 2)
  sns.boxplot(x=df1[col])
  plt.show()


# **Taking count of outliers and its Percentage after Data Transformations**
# 

# In[129]:


def find_outliers_IQR(df1):
    q1=df1.quantile(0.25)
    q3=df1.quantile(0.75)
    IQR=q3-q1
    outliers = df1[((df1<(q1-1.5*IQR)) | (df1>(q3+1.5*IQR)))]
    
    return outliers

for feature in numerical:
    outliers = find_outliers_IQR(df1[feature])

    print(feature)
    print('number of outliers: '+ str(len(outliers)))
    print('max outlier value: '+ str(outliers.max()))
    print('min outlier value: '+ str(outliers.min()))
    print('% of outliers: '+ str(len(outliers)/(len(df1[feature]))*100))
    print('\n')


# In[130]:


def detect_outliers(df1):
    outliers = {}
    for column in df1:
        # Calculate the IQR for the column
        Q1 = df1[column].quantile(0.25)
        Q3 = df1[column].quantile(0.75)
        IQR = Q3 - Q1

        # Identify the outliers using the IQR method
        outliers[column] = df1[(df1[column] < (Q1 - 1.5 * IQR)) | (df1[column] > (Q3 + 1.5 * IQR))].index

        # Print the number of outliers in the column
        print(f"{column}: {len(outliers[column])} outliers")

    # Check for duplicate outliers
    duplicate_outliers = []
    for column in outliers:
        for i in range(len(outliers[column])):
            index = outliers[column][i]
            for j in range(i+1, len(outliers[column])):
                if index == outliers[column][j]:
                    duplicate_outliers.append(index)
    if len(duplicate_outliers) > 0:
        print(f"Duplicate outliers found: {set(duplicate_outliers)}")
    else:
        print("No duplicate outliers found")

    return outliers


# In[131]:


df1


# In[132]:


df1_out=df1.copy()
df1_out.reset_index(inplace=True)
df1_out.index


# In[133]:


df1_out.columns


# In[134]:


df1_out['year'] = df1_out['time'].apply(lambda x:x.year)
df1_out['month'] = df1_out['time'].apply(lambda x:x.month)
df1_out['day'] = df1_out['time'].apply(lambda x:x.day)
##df1_out['weekday'] = df1_out['time'].apply(lambda x:x.weekday)
##df1_out['weekofyear'] = df1_out['time'].apply(lambda x:x.weekofyear)
df1_out['hour'] = df1_out['time'].apply(lambda x:x.hour)
df1_out['minute'] = df1_out['time'].apply(lambda x:x.minute)
df1_out


# **Splitting the dataframe based on seasons**

# In[135]:


def Season(month):
    if month in [12,1,2]:
        season = 'Winter'
    elif month in [3,4,5]:
        season = 'Spring'
    elif month in [6,7,8]:
        season = 'Summer'
    elif month in [9,10,11]:
        season = 'Autumn'            
    else:
        season='X'
            
    return season


# In[136]:


df1_out['season']= df1_out['month'].apply(Season)
for feature in numerical:
    plt.figure(figsize=(15,10))
    sns.boxplot(x='season', y=feature, data=df1_out)
    plt.title(feature)
    plt.show()


# In[137]:


for feature in numerical:
    plt.figure(figsize=(15,10))
    sns.boxenplot(x='season', y=feature, data=df1_out)
    plt.title(feature)
    plt.show()


# In[138]:


datause1=df1.iloc[:,0].values
#fig,ax=plt.subplots(figsize=(15,10))
plt.rcParams['figure.figsize'] = (14, 9)
seasonal_decompose(df1[['House overall']]).plot()
result = adfuller(datause1)
plt.show()


# In[139]:


datause=df1.iloc[:,0].values
#fig,ax=plt.subplots(figsize=(15,10))
plt.rcParams['figure.figsize'] = (14, 9)
seasonal_decompose(df1[['Solar']]).plot()
result = adfuller(datause)
plt.show()


# In[140]:


df1_prescale=df1.drop(['House overall','Solar'],axis=1)


# In[141]:


df1_prescale


# In[142]:


df1_prescale.index


# In[143]:


from sklearn.preprocessing import StandardScaler
stdsclr=StandardScaler()
X2=stdsclr.fit_transform(df1_prescale)


# In[144]:


(X2)


# In[145]:


X2=pd.DataFrame(X2,columns=[df1_prescale.columns])


# In[146]:


X2


# In[147]:


X2.describe().transpose()


# In[148]:


df1.describe().transpose()


# In[149]:


df1


# In[150]:


df1.info


# In[151]:


X2.columns=['Dishwasher', 'Home office', 'Fridge', 'Garage door',
        'Well', 'Microwave', 'Living room', 'Furnace',
       'Kitchen', 'temperature', 'humidity', 'pressure', 'windSpeed',
       'dewPoint']


# In[152]:


x = X2
y = df1['House overall']


# In[153]:


x.set_index(df1.index,inplace=True)


# In[154]:


x.index


# In[155]:


df1['House overall']


# In[156]:


X2.columns


# In[157]:


dataframe = pd.concat([x, y], axis=1)
dataframe


# In[158]:


new_csv_file = 'SmartHome.csv'
dataframe.to_csv(new_csv_file, index=False)


# ### Random forset

# In[159]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=52)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=52)
rf_model = rf_reg.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

print('MSE is ', mean_squared_error(y_test, y_pred_rf))
print('R Squared value :', r2_score(y_test, y_pred_rf))


# In[160]:


from sklearn.model_selection import cross_val_score, KFold
kfold_validator = KFold(5) 
cv_rf = cross_val_score(rf_model, x, y, cv = kfold_validator)
print("Validation result : ", cv_rf) 
print("\nAverage of Cross validation result ", np.mean(cv_rf))


# In[161]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def forecast_ts(df1, tt_ratio):
    X = df1.values
    size = int(len(X) * tt_ratio)
    train, test = X[:size], X[size:]
    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('progress:%', round(100 * (t / len(test))), '\t predicted=%f, expected=%f' % (yhat, obs), end="\r")

    error = mean_squared_error(test, predictions)
    print('\n Test MSE: %.3f' % error)
    print('\n Test MAE: ', mean_absolute_error(test, predictions))
    print('\n Test RMSE: ', np.sqrt(error))
    print('\n Test MPAE: ', mean_absolute_percentage_error(test, predictions))
    

    plt.rcParams["figure.figsize"] = (25, 10)
    preds = np.append(train, predictions)
    plt.plot(preds, color='green', linewidth=3, label="Predicted Data")
    plt.plot(df1.values, color='blue', linewidth=2, label="Original Data")
    plt.axvline(x=int(len(df1) * tt_ratio) - 1, linewidth=5, color='red')
    plt.legend()
    plt.show()

# Sample usage
col = 'House overall'
data = df1[col].resample('W').mean()

tt_ratio = 0.80 # Train to Test ratio
forecast_ts(data, tt_ratio)


# In[162]:


# splitting df1 as energy and weather data

energy_data2 = df1[['House overall', 'Dishwasher', 'Home office',
       'Fridge',  'Garage door',  'Well', 'Microwave',
       'Living room', 'Furnace', 'Kitchen']]

weather_data2 = df1[['temperature', 'humidity','pressure', 'windSpeed','dewPoint']]

energy_data2.index = pd.to_datetime(energy_data2.index)
weather_data2.index = pd.to_datetime(weather_data2.index)


# In[163]:


#resampling 

energy_per_hour2 = energy_data2.resample('H').sum()

energy_per_day2 = energy_data2.resample('D').sum()
energy_per_month2 = energy_data2.resample('M').sum()
energy_per_week2 = energy_data2.resample('W').sum()


weather_per_hour2 = weather_data2.resample('H').mean()
weather_per_day2 = weather_data2.resample('D').mean()
weather_per_month2 = weather_data2.resample('M').mean()
weather_per_week2 = weather_data2.resample('W').mean()


# In[164]:


#Plot ACF for energy 'gen'

print("Autocorrelation for 'Solar' = ", df1['Solar'].autocorr())
fig = plot_acf(df1['Solar'], lags=40, title="Autocorrelation for energy 'gen' per day")
plt.show()


# In[165]:


#Plot ACF for 'House overall'

print("Autocorrelation for 'House overall' = ", df1['House overall'].autocorr())
fig = plot_acf(df1['House overall'], lags=40, title="Autocorrelation for energy 'House overall' per day")
plt.show()


# ## **Anomaly detection**

# In[166]:


data = energy_per_day2.filter(items=['House overall'])
df = data

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(data.values.reshape(-1, 1))
xx = np.linspace(data.min(), data.max(), len(data)).reshape(-1,1)

df['scores']=isolation_forest.decision_function(df[['House overall']])
df['anomaly']=isolation_forest.predict(df[['House overall']])
df.head


# In[167]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# In[168]:


# Distribution of use energy per day
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Used Energy per Day')
plt.show();


# In[169]:


# Energy Per week anomaly detection
data = energy_per_week2.filter(items=['House overall'])
df = data

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(data.values.reshape(-1, 1))
xx = np.linspace(data.min(), data.max(), len(data)).reshape(-1,1)

df['scores']=isolation_forest.decision_function(df[['House overall']])
df['anomaly']=isolation_forest.predict(df[['House overall']])
df


# In[170]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# In[171]:


# Distribution of use energy per week
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Used Energy per Week')
plt.show();


# In[172]:


# Energy Per month anomaly detection
data = energy_per_month2.filter(items=['House overall'])
df = data

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(data.values.reshape(-1, 1))
xx = np.linspace(data.min(), data.max(), len(data)).reshape(-1,1)

df['scores']=isolation_forest.decision_function(df[['House overall']])
df['anomaly']=isolation_forest.predict(df[['House overall']])
df.head(5)


# In[173]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# In[174]:


# Distribution of use energy per month
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Used Energy per Month')
plt.show();


# In[175]:


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

l1=[]
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(25,5))
    plt.title("Moving average (window size = {})".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

  # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
               
      
      # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, marker="o",markerfacecolor='red', markersize=12)
            
            #l1[:n_samples]=[upper_bond]
            #l2[:n_samples]=[lower_bond]
            print(anomalies.values)
            l1.append(anomalies.values)
            
            
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

n_samples = 8399 # 1 month
data = energy_per_hour2.filter(items=['House overall'])
plotMovingAverage(data[:n_samples], window=12)


# In[176]:


plotMovingAverage(data[:n_samples], window=24, plot_intervals=True ,plot_anomalies=True)


# In[177]:


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

l1=[]
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(25,5))
    plt.title("Moving average (window size = {})".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

  # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
         
        
      
      # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, marker="o",markerfacecolor='red', markersize=12)
            
            
            print(anomalies.values)
            l1.append(anomalies.values)
            
            
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

n_samples = 24*30 # 1 month
data = energy_per_day2.filter(items=['House overall'])
plotMovingAverage(data[:n_samples], window=24, plot_intervals=True ,plot_anomalies=True)


# In[178]:


n_samples = 24*30 # 1 month
for i in energy_per_day2.columns:
  data = energy_per_day2[i]
  plotMovingAverage(data[:n_samples], window=6 )


# In[179]:


data = energy_per_day.filter(items=['House overall'])
plotMovingAverage(data[:n_samples], window=6)


# In[180]:


plotMovingAverage(data[:n_samples], window=24, plot_intervals=True ,plot_anomalies=True)


# In[181]:


#Autocorrelation plot is used to find the AR parameter p of the ARIMA model
print("Autocorrelation for 'use' = ", energy_per_day2['House overall'].autocorr())
fig= plot_acf(energy_per_day2['House overall'], lags=50)
plt.show()


# In[182]:


plot_pacf(energy_per_day2['House overall'], lags=40)


# In[183]:


plot_pacf(energy_per_day2['House overall'], lags=40)


# In[184]:


print("Autocorrelation for {}= ",energy_per_day2['House overall'].autocorr())
fig= plot_acf(energy_per_day2['House overall'], lags=40)
plt.show()


# In[185]:


#Autocorrelation plot is used to find the AR parameter p of the ARIMA model

for i in df1.columns:
  print("Autocorrelation for {}= ".format(i),df1[i].autocorr())
  array = np.array(df1[i]).reshape(-1,1)
  fig= plot_acf(df1[i], lags=40)
  plt.show()


# In[187]:


from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error
import time 
for i in df1.columns:
  array = np.array(df1[i]).reshape(-1,1)
  size = int(len(array) * 0.70)
  train, test = array[0:size], array[size:len(array)]
  history = train.flatten().tolist()
  predictions = list()
  start_time= time.time()
  for t in range(len(test)):    
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t][0]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

  error = mean_squared_error(test, predictions)
  end_time= time.time()
  time_taken= end_time-start_time
  print('Time taken to train the model in seconds= ',time_taken)
  print('Test MSE: %.3f' % error)
  # plot
  plt.plot(test)
  plt.plot(predictions, color='red')
  plt.show()


# In[191]:


home_arima_df=df1[0:1000]


# In[192]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import time
for i in home_arima_df.columns:
    array= np.array( home_arima_df[i]).reshape(-1,1)
    size = int(len(array) * 0.70)
    train, test = array[0:size], array[size:len(array)]
    history = [x for x in train]
    predictions = list()
    start_time= time.time()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    end_time= time.time()
    time_taken= end_time-start_time
    print('Time taken to train the model in seconds= ',time_taken,'-{}'.format(i))
    print('Test MSE: %.3f' % error)
# plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


# ThymeBoost

# In[195]:


from ThymeBoost import ThymeBoost as tb
boosted_model = tb.ThymeBoost(approximate_splits=True,
                              n_split_proposals=25,
                              verbose=1,
                              cost_penalty=.001)


# In[197]:


list = df1['House overall'].tolist()
output = boosted_model.fit(list,
                           trend_estimator='linear',
                           seasonal_estimator='fourier',
                           seasonal_period=25,
                           split_cost='mse',
                           global_cost='maicc',
                           fit_type='global')
boosted_model.plot_results(output)
boosted_model.plot_components(output)


# In[198]:


boosted_model = tb.ThymeBoost()
output = boosted_model.detect_outliers(list,
                                       trend_estimator='linear',
                                       seasonal_estimator='fourier',
                                       seasonal_period=25,
                                       global_cost='maicc',
                                       fit_type='global')
boosted_model.plot_results(output)
boosted_model.plot_components(output)


# In[199]:


weights = np.invert(output['outliers'].values) * 1
output = boosted_model.fit(list, trend_estimator='linear',
                           seasonal_estimator='fourier',
                           seasonal_period=25,
                           global_cost='maicc',
                           fit_type='global',
                           seasonality_weights=weights)
boosted_model.plot_results(output)
boosted_model.plot_components(output)


# In[200]:


output = boosted_model.fit(list,
                           trend_estimator='linear',
                           seasonal_estimator='fourier',
                           seasonal_period=25,
                           global_cost='maicc',
                           fit_type='global',
                           seasonality_weights='regularize')
boosted_model.plot_components(output)
boosted_model.plot_results(output)


# In[201]:


output


# In[ ]:




