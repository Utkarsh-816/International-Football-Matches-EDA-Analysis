#!/usr/bin/env python
# coding: utf-8

# In[23]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


# In[3]:


df=pd.read_csv(r"C:\Users\user\Downloads\international_matches (1).csv")


# In[4]:


df


# In[5]:


#checking tot row and column in data
df.shape


# In[7]:


#Checking null values
df.isnull().sum()


# In[13]:


df.duplicated


# In[14]:


df.columns


# In[15]:


#checking data type
df.dtypes


# In[16]:


# as we can see the datatype of attribute "date" is object so we need to chage it into "datetime"
df["date"]=pd.to_datetime(df["date"])
type(df["date"])


# In[17]:


#Checking categorical attributes
cat_features=df.select_dtypes(include="object")
print("the shape of categorical feature is:",cat_features.shape)
print("the name of cat features are :",cat_features.columns)


# In[25]:


label_encoder=LabelEncoder()
df_encoded=df.copy()


# In[22]:


df_encoded["home_team"]=label_encoder.fit_transform(df_encoded["home_team"])
df_encodd["home_team"].astype(float)


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


label_encoder=LabelEncoder()
df_encoded=df.copy()


# In[29]:


df_encoded["home_team"]=label_encoder.fit_transform(df_encoded["home_team"])
df_encoded["home_team"].astype(float)


# In[30]:


df_encoded["away_team"]=label_encoder.fit_transform(df_encoded["away_team"])
df_encoded["away_team"].astype(float)


# In[31]:


df_encoded["away_team_continent"]=label_encoder.fit_transform(df_encoded["away_team_continent"])
df_encoded["away_team_continent"].astype(float)


# In[32]:


df_encoded["tournament"]=label_encoder.fit_transform(df_encoded["tournament"])
df_encoded["tournament"].astype(float)


# In[33]:


df_encoded["city"]=label_encoder.fit_transform(df_encoded["city"])
df_encoded["city"].astype(float)


# In[34]:


df_encoded["shoot_out"]=label_encoder.fit_transform(df_encoded["shoot_out"])
df_encoded["shoot_out"].astype(float)


# In[35]:


df_encoded["home_team_result"]=label_encoder.fit_transform(df_encoded["home_team_result"])
df_encoded["home_team_result"].astype(float)


# In[36]:


df.head()


# In[37]:


df.dtypes


# In[38]:


#checking numerical attributes
Numerical_df=df.select_dtypes(include="number")
print("shape of numerical feature is:",Numerical_df.shape)
print("name of numerical feature is:",Numerical_df.columns)


# In[40]:


#checking boolean attributes
bool_df=df.select_dtypes(include="bool")
print("shape of boolean feature is:",bool_df.shape)
print("name of boolean feature is:",bool_df.columns)


# In[41]:


df_encoded["neutral_location"]=label_encoder.fit_transform(df_encoded["neutral_location"])
df_encoded["neutral_location"].astype(float)


# In[43]:


#checking date time attributes
dtime_df=df.select_dtypes(include="datetime")
print("shape of datetime feature is:",dtime_df.shape)
print("name of datetime feature is:",dtime_df.columns)


# In[44]:


df.describe()


# In[ ]:


# as we can see we have applied descriptive stat on our dataset here.
#in this description we can see the following aspects
1-count
2-mean
3-standard deviation
4-Quartiles
5-Range


# In[45]:


df_corr=pd.DataFrame(df)
print(df_encoded.corr())


# In[ ]:


# we can see that last 8 column showed very poor correlation due to more than 60 per missing values
#we can assume it is safe to drop them without affecting the rest of dataset


# In[79]:


df.drop(columns=["home_team_goalkeeper_score","away_team_goalkeeper_score","home_team_mean_defense_score","home_team_mean_offense_score","home_team_mean_midfield_score","away_team_mean_defense_score","away_team_mean_offense_score","away_team_mean_midfield_score"],inplace=True)


# In[47]:





# In[ ]:


#Interpretation through data visualization


# In[48]:


#comparing home team continent and home team result
plt.figure(figsize=(10,5))
sns.set_style("darkgrid")
sns.countplot(x="away_team_continent",hue="home_team_result",data=df,palette="deep")
plt.xlabel("Away Team Continent")
plt.ylabel("Count")
plt.show()


# In[ ]:


#interpretation
1-south america is the most difficult continent to visit by teams of other continents
2-mostly team who visit north america return as winners
3-despite being the smallest continents oceania still proves to be harder for visiting teams then other mighty football continents
such as africa or north amaerica


# In[ ]:


#insights of matches drawn


# In[49]:


#Getting insights for draw result
Draws=df[df["home_team_result"]=="Draw"]
Draws.shape


# In[53]:


X=Draws["country"]
y=Draws["home_team_result"]


# In[60]:



sns.countplot(data=Draws,x=X,hue=y)
plt.xlabel=("country")
plt.ylabel("home_team_result")
plt.title("home team result vs country")
plt.show()


# In[ ]:


#there are 217 tot countries
max Draws we got around 218.


# In[61]:


import plotly.express as px


# In[63]:


fig=px.scatter(df,x="home_team",y="home_team_result",size="home_team_fifa_rank",color="country",size_max=55,animation_frame="country",
              animation_group="home_team_result")
fig.show()


# In[ ]:


# we plotted a scatterplot using plotly to see the following things:
1-do the fifa rank affect win,loss and draw ratio
2-do the city where match is being played affect the win,loss and draw ratio of a team


# In[138]:


import matplotlib.pyplot as plt
df.home_team_continent.value_counts()
plt.bar(list(df["home_team_continent"].value_counts()[0:5].keys()) ,list(df["home_team_continent"].value_counts()[0:5]))
plt.title('continent Summary',fontsize=20)
plt.ylabel('Matches Played',fontsize=15)
plt.xticks(rotation=30)
plt.show()


# In[76]:


df.home_team_continent.value_counts()


# In[ ]:


# interpretation
1-most matches played in europe
2-africa has a very big potential of football talent


# In[84]:


df.tournament.value_counts()                                                            


# In[ ]:


sns.countplot(df["home_team_continent"].value_counts().head(5))


# In[141]:



sns.catplot(data=df,x="home_team_result",kind="count")


# In[86]:


df.head(5)


# In[108]:





# In[109]:


sns.catplot(data=df.head(5),x="away_team",y="away_team_fifa_rank",kind="bar")


# In[111]:


sns.catplot(data=df.tail(5),x="home_team",y="home_team_fifa_rank",kind="bar")


# In[113]:


sns.catplot(data=df.tail(5),x="away_team",y="away_team_fifa_rank",kind="bar")


# In[116]:





# In[136]:


import matplotlib.pyplot as plt
df.country.value_counts()
plt.bar(list(df["country"].value_counts()[0:5].keys()) ,list(df["country"].value_counts()[0:5]))
plt.title('country matches',fontsize=20)
plt.ylabel('Matches Played',fontsize=15)
plt.xticks(rotation=30)
plt.show()


# In[134]:


import matplotlib.pyplot as plt
df.tournament.value_counts()
plt.bar(list(df["tournament"].value_counts()[0:5].keys()) ,list(df["tournament"].value_counts()[0:5]))
plt.title('Tournament Summary',fontsize=20)
plt.ylabel('Matches Played',fontsize=15)
plt.xticks(rotation=30)
plt.show()


# In[ ]:


#interpretation
1-most matches played are friendly
2-AFC asian cup has very little effect on decision of world football


# In[ ]:


#Conclusion
1-Most Enthusiastic continents of football world are Europe and Africa
2-Most competitive continent is south America
3-Home teams always have an upper edge
4-Africa have to focus on football industry
5-Europe has a lot of earning potential for bushiness related to football industry

