#!/usr/bin/env python
# coding: utf-8

# In[1]:


#setting working dorectory to below path
import os
os.chdir('C:\\Users\\ASUS\\desktop\\ALL\\Hackathon\\MAchineHAck\\Dataset')


# In[3]:


#importing initially required libraries for data exploration
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


#reading df into a data frame named df
df = pd.read_csv('Train.csv')


# In[5]:


#having a look at the data by using inbuilt pandas function 
df.head()


# In[6]:


df.isna().sum()


# In[7]:


#found out holiday column has NaN so replacing it with zero
df['Holiday'] = df['Holiday'].fillna(0)


# In[8]:


#checking the data distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Traffic_Vol'], bins=30, edgecolor='black')  # 30 bins
plt.title('Distribution of Traffic Volume')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')
plt.show()


# In[9]:


#checking for data types
print(df.dtypes)


# In[10]:


df.head()


# In[11]:


#based on abobe analhysis changing TimeStamp into datetime format from object
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])


# In[12]:


#Extracting year, month, day and hour from Timestamp to create new parameter crucial for model
df['Year'] = df['TimeStamp'].dt.year
df['Month'] = df['TimeStamp'].dt.month
df['Day'] = df['TimeStamp'].dt.day
df['Hour'] = df['TimeStamp'].dt.hour


# In[13]:


#Dropping date column whihc looks redundant as we have timestamp to asiist the purpose
df = df.drop('Date', axis=1)


# In[14]:


#Dividing days into weekday and weekend to try and boost model performance
df['dayofweek'] = df['TimeStamp'].dt.dayofweek 
df['is_weekend'] = df['TimeStamp'].dt.dayofweek >= 5  


# In[15]:


#Dropping TimeStamp column as we have created derived attributes from it and does not require it anymore for model purpose
df = df.drop('TimeStamp', axis= 1)
df.head()


# In[118]:


df.Weather_Desc.value_counts()


# In[16]:


#Dropping Weather_desc as we are using weather column to make derived attributes to avoid potential repetition
df = df.drop('Weather_Desc', axis = 1)


# In[17]:


#converting categorical data from weather into numerical using one hot encoding
df = pd.get_dummies(df, columns = ['Weather'])


# In[23]:


#converting the data type of newly created categorical columns into integer.
df = df.astype(int)


# In[27]:


#checking the data again to test the dataset for readiness
df.head()


# In[ ]:


#Checking for potential outliers in Target column/variable. 
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Traffic_Vol'])
plt.title('Boxplot of Traffic Volume')
plt.show()


# In[35]:


df = df.drop('Weather_Sudden windstorm', axis = 1)
df = df.drop('Weather_Snowfall', axis = 1)


# In[39]:


df.head()


# In[41]:


df.columns


# In[78]:


df_test.columns


# In[202]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor with adjusted hyperparameters
model = RandomForestRegressor(
    n_estimators=33,            # Number of trees
    max_depth=35,                # Limit tree depth to reduce complexity
    min_samples_split=5,        # Require more samples to split an internal node
    min_samples_leaf=2,          # Minimum samples per leaf
    max_features='sqrt',         # Limit the number of features per split
    random_state=42
)

# Fit the model
model.fit(X_train, y_train)

# Predict and evaluate
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse}")


# In[204]:


y_train_pred = model.predict(X_train)

# Calcualting the RMSE as asked
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"train RMSE: {rmse}")


# In[43]:


df_test = pd.read_csv('Test.csv')


# In[45]:


df_test['Holiday'] = df_test['Holiday'].fillna(0)


# In[47]:


df_test.head()


# In[49]:


df_test.isna().sum()


# In[51]:


df_test['Holiday'] = df_test['Holiday'].fillna(0)


# In[53]:


df_test = df_test.drop('Date', axis=1)


# In[55]:


df_test['TimeStamp'] = pd.to_datetime(df_test['TimeStamp'])


# In[57]:


df_test['Year'] = df_test['TimeStamp'].dt.year
df_test['Month'] = df_test['TimeStamp'].dt.month
df_test['Day'] = df_test['TimeStamp'].dt.day
df_test['Hour'] = df_test['TimeStamp'].dt.hour


# In[59]:


df_test['dayofweek'] = df_test['TimeStamp'].dt.dayofweek 
df_test['is_weekend'] = df_test['TimeStamp'].dt.dayofweek >= 5  


# In[61]:


df_test.head()


# In[63]:


df_test = df_test.drop('Weather_Desc', axis = 1)


# In[65]:


df_test = pd.get_dummies(df_test, columns=['Weather'])


# In[67]:


df_test.head()


# In[72]:


df_test = df_test.drop('TimeStamp', axis = 1)


# In[74]:


df_test.shape


# In[76]:


df.shape


# In[92]:


df_test.columns


# In[98]:


df_test = df_test.drop('Traffic_Vol', axis = 1)


# In[206]:


y_test_pred = model.predict(df_test)


# In[236]:


y_test_pred[:6]


# In[238]:


df_subm = pd.read_csv('Submission.csv')


# In[240]:


df_subm.head()


# In[242]:


df_subm['Traffic_Vol'] = y_test_pred


# In[244]:


df_subm.head()


# In[248]:


df_subm.to_csv('Submission_2.csv')


# In[ ]:




