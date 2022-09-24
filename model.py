#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Feature information ##
## This function provides description for variables in dataset ##
import pandas as pd
data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
    
feat_info('mort_acc')


# In[ ]:


## Import libraries and data ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('lending_club_loan_two.csv')
df.info()


# ## DATA ANALYSIS

# In[ ]:


## Countplot of prediction variable ##
sns.countplot(x='loan_status',data=df)

## Histogram of loan amount ##
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)

## Correlation of variable and their heatmap ##
df.corr()

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# In[ ]:


## Correlation between loan_amnt and installment #
feat_info('installment')
feat_info('loan_amnt')

sns.scatterplot(x='installment',y='loan_amnt',data=df,)


# In[ ]:


## Relation between loan_amnt and loan_status ##
sns.boxplot(x='loan_status',y='loan_amnt',data=df)

df.groupby('loan_status')['loan_amnt'].describe()


# In[ ]:


## Showing the relation of loan_amnt with various grades and sub_grades ##
sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())

sns.countplot(x='grade',data=df,hue='loan_status')

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# In[ ]:


## Creating a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off" ##
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]

## A bar plot showing the correlation of the numeric features to the new loan_repaid column ##
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# ## DATA PREPROCESSING

# In[ ]:


## Finding missing data ##
df.isnull().sum()
100* df.isnull().sum()/len(df)

## Analysing emp_title ##
df['emp_title'].value_counts()

## Removing emp_title as a dummy variable ##
df = df.drop('emp_title',axis=1)

## Analysing emp_length ##
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

## Percentage of charge of per category ##
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp

emp_len.plot(kind='bar')

## Removing emp_length ##
df = df.drop('emp_length',axis=1)

## title and purpose are similar ##
df['title'].head(10)
df['purpose'].head(10)
df = df.drop('title',axis=1)


# In[ ]:


## Mortage_acc ##
df.corr()['mort_acc'].sort_values()
df.groupby('total_acc').mean()['mort_acc']

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

## fill in the missing mort_acc values based on their total_acc value ##

def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
    
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()

## Removing the rows with missing value data ##
df = df.dropna()


# In[ ]:


## Categorical and dummy variables ##
df.select_dtypes(['object']).columns

## Term to integer ##
df['term'] = df['term'].apply(lambda term: int(term[:3]))

## Keep just grade or sub_grade ##
df = df.drop('grade',axis=1)

## Converting sub_grades to dummy variables ##
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

## Converting these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variable ##
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

## Home_owenership analysis ##
df['home_ownership'].value_counts()

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

## New column Zip_code ##
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

df = df.drop('issue_d',axis=1)

## Creating an new column 'earliest_cr_year' inclusing only year ##
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# ## MODEL CREATION

# In[ ]:


## Split test and train data ##
from sklearn.model_selection import train_test_split

df = df.drop('loan_status',axis=1)
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

## Normalizing the data ##
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


## Model Creation ##
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

## 78--> 39 --> 19 --> 1 ##
model = Sequential()

## Input layer ##
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

## Hidden layer ##
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

## Hidden layer ##
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

## Output layer ##
model.add(Dense(units=1,activation='sigmoid'))

## Compile model ##
model.compile(loss='binary_crossentropy', optimizer='adam')

## Training ##
model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )

## Save model ##
from tensorflow.keras.models import load_model
model.save('project_model.h5')  


# ## MODEL EVALUATION

# In[ ]:


## Plot out the validation loss versus the training loss ##
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

## Classification report ##
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)

## Offer a loan or not ##
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer

model.predict_classes(new_customer.values.reshape(1,78))
df.iloc[random_ind]['loan_repaid']

