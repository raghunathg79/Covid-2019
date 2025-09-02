#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
get_ipython().run_line_magic('matplotlib', 'inline')
import os

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

get_ipython().system('pip install folium')
import folium


# In[3]:


Covid19=pd.read_csv("C:\\Users\\RAGHUNATH GANESAN\\Documents\\DATA SCIENCE\\GIT_HUB\\Bing-COVID-19-Data\\data\\Bing-COVID19-Data.csv")


# In[4]:


Covid19.describe()


# In[5]:


Covid19.head()


# In[6]:


Covid19.tail()


# In[9]:


Covid19.corr()


# In[10]:


Covid19.info()


# In[11]:


plt.scatter(x=Covid19['Country_Region'],y=Covid19['Confirmed'])
plt.title("Covid19_Countrywise")
plt.xlabel("Country_Region")
plt.ylabel("Confirmed")
plt.figure(figsize=(15,8))


# In[25]:


world = Covid19.groupby('Country_Region')['Confirmed','Deaths','Recovered'].sum().reset_index
world


# In[12]:


Covid19.loc[:, ['Country_Region','AdminRegion1','AdminRegion2','Latitude','Longitude']]


# In[13]:


df1=(Covid19[Covid19['AdminRegion1'] == "Karnataka"])


# In[14]:


df1.drop(df1.index[df1['AdminRegion2'] == 'NaN'], inplace = True)


# In[15]:


df1


# In[16]:


df1.to_csv('Karnataka_160422.csv')


# In[17]:


Map=df1.loc[:, ['Updated','Country_Region','AdminRegion1','AdminRegion2','Latitude','Longitude','Confirmed','Deaths','Recovered']]


# In[18]:


Map


# In[16]:


df2=(Covid19[Covid19['AdminRegion2'] == "Bangalore Urban district"])


# In[17]:


df2


# In[18]:


#Global_Map=folium.Map()
#Global_Map


# In[19]:


#m = folium.Map(location=[df1.Latitude.mean(),df1["Longitude"].mean()])


# In[20]:


#for index, location_info in Covid19_location.iterrows():
    #folium.Marker([location_info["Latitude"],location_info["Longitude"]],popup=location_info["AdminRegion2"]).add_to(m)
    #print([location_info["Latitude"],location_info["Longitude"],location_info["AdminRegion2"]])


# In[21]:


#m


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
get_ipython().run_line_magic('matplotlib', 'inline')
import os

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[23]:


Location = ("C:\\Users\\RAGHUNATH GANESAN\\Documents\\DATA SCIENCE\\GIT_HUB\\Bing-COVID-19-Data\\data\\Bing-COVID19-Data.csv")


# In[24]:


Covid19_locations=pd.read_csv(Location)


# In[25]:


Covid19_locations


# In[26]:


Covid19_locations=Covid19_locations[["Country_Region","AdminRegion1","AdminRegion2","Latitude","Longitude","Confirmed","ConfirmedChange","Deaths","DeathsChange","Recovered","RecoveredChange"]]


# In[27]:


Covid19_locations.isnull().sum()


# In[28]:


Covid19_locations["AdminRegion1"].dropna()


# In[29]:


Covid19_locations["AdminRegion2"].dropna()


# In[30]:


Covid19_locations["Latitude"].dropna()


# In[31]:


Covid19_locations["Longitude"].dropna()


# In[32]:


Covid19_locations.drop(Covid19_locations.index[Covid19_locations['Country_Region'] == 'Worldwide'], inplace = True)


# In[33]:


Covid19_locations["ConfirmedChange"].fillna(Covid19_locations["ConfirmedChange"].mean(),inplace=True)


# In[34]:


Covid19_locations.isnull().sum()


# In[35]:


Covid19_locations["DeathsChange"].fillna(Covid19_locations["DeathsChange"].mean(),inplace=True)


# In[36]:


Covid19_locations["Deaths"].fillna(Covid19_locations["Deaths"].mean(),inplace=True)


# In[37]:


Covid19_locations["Recovered"].fillna(Covid19_locations["Recovered"].mean(),inplace=True)


# In[38]:


Covid19_locations["RecoveredChange"].fillna(Covid19_locations["RecoveredChange"].mean(),inplace=True)


# In[39]:


Covid19_locations


# In[40]:


Covid19_location=(Covid19_locations[Covid19_locations['AdminRegion1'] == "Karnataka"])


# In[41]:


Covid19_locations.to_csv=('Covid19_Karnataka.csv')


# In[42]:


Covid19_location.Latitude.mean()


# In[43]:


Covid19_location.Longitude.mean()


# # Modelling 

# #df["Recovered"].fillna(df["Recovered"].mean,inplace = True)
# #df["Recovered"] = df["Recovered"].astype(int(round="Recovered"))

# In[44]:


Covid19_locations["Deaths"].fillna(Covid19_locations["Deaths"].mean,inplace = True)
#Covid19_locations["Recovered"] = Covid19_locations["Recovered"].astype(int(round="Recovered"))


# In[45]:


Covid19_locations.isnull().sum()


# In[46]:


Covid19_locations["Country_Region"].value_counts()


# In[47]:


df2= Covid19_locations["AdminRegion1"].value_counts()


# In[48]:


Covid19_locations["AdminRegion2"].value_counts()


# In[49]:


Covid19_locations=(Covid19[Covid19['AdminRegion2'] == "Bangalore Urban district"])


# In[50]:


Covid19_locations["Deaths"]=Covid19_locations["Deaths"].astype('int64')


# In[51]:


#plt.scatter(Covid19_locations[[Covid19['AdminRegion2'] =="Bangalore Urban district"],Covid19_locations["Deaths"]])


# In[52]:


Covid19_locations["AdminRegion2"].value_counts()


# In[53]:


plt.scatter(x=Covid19_locations['Updated'],y=Covid19_locations['Deaths'])


# In[54]:


X=Covid19_locations[['Confirmed']]
Y=Covid19_locations[['Deaths','Recovered']]


# In[55]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[56]:


linr=LinearRegression()


# In[57]:


linr.fit(X_train,Y_train)


# In[58]:


Y_predict=linr.predict(X_test)


# In[59]:


r_squared=metrics.r2_score(Y_test,Y_predict)
r_squared


# In[ ]:




