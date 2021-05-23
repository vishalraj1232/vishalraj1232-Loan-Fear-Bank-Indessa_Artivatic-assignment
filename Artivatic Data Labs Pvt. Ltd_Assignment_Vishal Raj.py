#!/usr/bin/env python
# coding: utf-8

# # Artivatic Data Labs Pvt. Ltd

# Problem Statement:
# The Bank Indessa has not done well in the last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose the confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.
# After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.
# This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a loan of $20000, along with the bank, the investors perform due diligence on the requested loan application. Keep this in mind while understanding data.
# In this challenge, you will help this bank by predicting the probability that a member will default

# By Vishal Raj
# raj.19@iitj.ac.in /vishal.vishalraj@gmail.com

# In[1]:


#Importing packages
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math


# ## Upload the training dataset

# In[2]:


dfTrain = pd.read_csv('train_indessa.csv')
dfTest = pd.read_csv('test_indessa.csv')


# In[3]:


dfTrain.head() 


# In[4]:


dfTest.head()


# Find the shape of the dataframe? and find out all columns are present in the dataframe?

# In[5]:


dfTrain.shape


# In[6]:


dfTest.shape


# In[7]:


dfTrain.columns


# In[8]:


dfTest.columns


# In[9]:


dfTrain.describe()


# In[10]:


dfTest.describe()


# In[11]:


dfTrain.info()


# In[12]:


dfTest.info()


# ## 2. Preprocessing the Dataset

# finding the distribution of our target variable?

# In[13]:


dfTrain['loan_status'].value_counts()


# ## 2.1 Term Feature
# We'll do some preprocessing for the term attribute.
# 
# Now, let's so some data cleaning. We'll remove months from term column.

# In[14]:


dfTrain['term'] = dfTrain['term'].str.replace('months', '') # Removes months
dfTrain['term'] = dfTrain['term'].str.replace(' ', "")      # Removes the space left out after removing months


# In[15]:



dfTest['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
dfTest['term'] = pd.to_numeric(dfTest['term'], errors='coerce')


# Let's check if we removed them and if the column just contains integers.

# In[16]:


dfTrain.term.unique() 


# In[17]:


dfTest.term.unique() 


# ## 2.2 Employment Length Feature
# We'll do some preprocessing for the emp_length attribute. We'll now clean the emp_length column in our dataframe.

# In[18]:



dfTrain['emp_length'].replace('n/a', '0', inplace=True)
dfTrain['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True) 
dfTrain['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True) 
dfTrain['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
dfTrain['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
dfTest['emp_length'].replace('n/a', '0', inplace=True)
dfTest['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)

# Convert it to numeric
dfTrain['emp_length'] = pd.to_numeric(dfTrain['emp_length'], errors='coerce')
dfTest['emp_length'] = pd.to_numeric(dfTest['emp_length'], errors='coerce')


# In[19]:


dfTrain.head() 


# In[20]:


dfTest.head() 


# 
# Let's have a look at the unique value distribution of the emp_length column.

# In[21]:


dfTrain['emp_length'].value_counts()


# In[22]:


dfTest['emp_length'].value_counts()


# Let's check if the column is cleaned and contains only distinct integers. We'll be dealing with nan values later.

# In[23]:


dfTrain.emp_length.unique()


# In[24]:


dfTest.emp_length.unique()


# How many nan values does the emp_length have?

# In[25]:


dfTrain.emp_length.isnull().sum() 


# In[26]:


dfTest.emp_length.isnull().sum() 


# ## 2.3 Last Week Pay Feature
# We'll now clean the last_week_pay column in our dataframe.

# In[27]:



dfTrain['last_week_pay'].replace(to_replace='th week', value='', regex=True, inplace=True)
dfTest['last_week_pay'].replace(to_replace='th week', value='', regex=True, inplace=True)
dfTrain['last_week_pay'].replace(to_replace='NA', value='', regex=True, inplace=True)
dfTest['last_week_pay'].replace(to_replace='NA', value='', regex=True, inplace=True)

# Convert it to numeric
dfTrain['last_week_pay'] = pd.to_numeric(dfTrain['last_week_pay'], errors='coerce')
dfTest['last_week_pay'] = pd.to_numeric(dfTest['last_week_pay'], errors='coerce')


# ## 2.4 Sub Grade Feature
# We'll do some preprocessing for the sub_grade attribute.

# In[28]:


print('Transform: sub_grade...')
dfTrain['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
dfTrain['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
dfTest['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
dfTrain['sub_grade'] = pd.to_numeric(dfTrain['sub_grade'], errors='coerce')
dfTest['sub_grade'] = pd.to_numeric(dfTest['sub_grade'], errors='coerce')

# Convert it to numeric
dfTrain['sub_grade'] = pd.to_numeric(dfTrain['sub_grade'], errors='coerce')
dfTest['sub_grade'] = pd.to_numeric(dfTest['sub_grade'], errors='coerce')


# In[29]:


dfTrain.head() 


# In[30]:


dfTest.head() 


# Let's have a look at the unique value distribution of the sub_grade column.

# In[31]:


dfTrain['sub_grade'].value_counts()


# In[32]:


dfTest['sub_grade'].value_counts()


# ## 3. Handling Missing Values
# Now, we'll see how many null values does each attribute have.

# In[33]:



dfTrain.isnull().sum()


# In[34]:



dfTest.isnull().sum()


# ## 3.1 Missing value imputation

# In[35]:


columns = ['term', 'loan_amnt', 'funded_amnt', 'last_week_pay', 'int_rate', 'sub_grade',
           'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
           'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 
           'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
for col in columns:
    print('Imputation with Median: %s' % (col))
    dfTrain[col].fillna(dfTrain[col].median(), inplace=True)  # Filling NaN values with median of each column present in columns.

  
num_cols = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
for col in num_cols:
    print('Imputation with Zero: %s' % (col))
    dfTrain[col].fillna(0, inplace=True)        ## Filling NaN values with 0 for each column present in columns.

print('Missing value imputation done for training data.')


# In[36]:


columns = ['term', 'loan_amnt', 'funded_amnt', 'last_week_pay', 'int_rate', 'sub_grade',
           'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
           'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 
           'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
for col in columns:
    print('Imputation with Median: %s' % (col))
    dfTest[col].fillna(dfTest[col].median(), inplace=True)  # Filling NaN values with median of each column present in columns.

  
num_cols = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
for col in num_cols:
    print('Imputation with Zero: %s' % (col))
    dfTest[col].fillna(0, inplace=True)        ## Filling NaN values with 0 for each column present in columns.

print('Missing value imputation done for testing data.')


# In[37]:


dfTrain.isnull().sum()


# In[38]:


dfTest.isnull().sum()


# In[39]:


dfTrain.replace([np.inf, -np.inf], np.nan, inplace=True)
dfTest.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[40]:


'''
Missing values imputation
'''
cols = ['term', 'loan_amnt', 'funded_amnt', 'last_week_pay', 'int_rate', 'sub_grade', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
for col in cols:
    print('Imputation with Median: %s' % (col))
    dfTrain[col].fillna(dfTrain[col].median(), inplace=True)
    dfTest[col].fillna(dfTest[col].median(), inplace=True)

cols = ['acc_now_delinq', 'batch_enrolled','emp_title','desc','title','delinq_2yrs','inq_last_6mths','pub_rec','verification_status_joint','total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
for col in cols:
    print('Imputation with Zero: %s' % (col))
    dfTrain[col].fillna(0, inplace=True)
    dfTest[col].fillna(0, inplace=True)

print('Missing value imputation done.')


# In[41]:


dfTest.isnull().sum()


# ## 4. Exploratory Data Analysis 

# In[42]:


import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## categorical attributes visualization

# ## 4.1 Loan Status Vs Grade

# In[43]:


sns.countplot(dfTrain['grade'])


# Overview of the above chart:
# 
# Most of the Loans (irrespective of Defaulter or Non-Defaulter) were taken by B grade.
# Least of the Loans (irrespective of Defaulter or Non-Defaulter) were taken by G grade.
# Top 3 loan takers are - B, C and A grade.

# ## 4.2 Home Ownership Vs Grade

# In[44]:


dfTrain.home_ownership.unique()


# In[45]:


sns.countplot(dfTrain["home_ownership"])


# In[46]:


dfTrain["home_ownership"]


# In[47]:



sns.countplot(x ="home_ownership",hue="grade", data = dfTrain)
plt.show()


# Overview of the above chart:
# 
# Most of the Home Ownerships were taken by B and C grade.
# Least of the Home Ownerships were taken by G grade.
# Top 3 - B, C and A grade.

# ## 4.3 Loan Status Vs Verification Status

# In[48]:


sns.countplot(dfTrain["verification_status"])


# Overview of the above chart:
# 
# Most of the Loans (irrespective of Defaulter or Non-Defaulter) were Not Verified.
# Least of the Loans (irrespective of Defaulter or Non-Defaulter) were Source Verified.
# If we consider Source Verified and Verified to be the same, then most of the loans were Verified but still quite a good amount of Loans (35.7%) were not Verified.

# ## 4.4 Loan Status Vs Term

# In[49]:


sns.barplot(x = "term", y = "loan_status", data = dfTrain)
plt.show()


# Overview of the above chart:
# 
# Most of the Loans (irrespective of Defaulter or Non-Defaulter) were taken for a term of 36 months.
# This clearly shows that most of the Customers likes a short term loan rather than a long term loan.

# ## 4.5 Loan Status Vs Employment Length 

# In[50]:


sns.barplot(x = "emp_length", y = "loan_status", data = dfTrain)
plt.show()


# Overview of the above chart:
# 
# Most of the Loans (irrespective of Defaulter or Non-Defaulter) were 5 years employed.
# Least of the Loans (irrespective of Defaulter or Non-Defaulter) were 6 years employed.

# ## 4.6 Purpose Vs loan status

# In[51]:


dfTrain.loan_status.unique()


# In[52]:


dfTrain.purpose.unique()


# In[53]:


sns.countplot(dfTrain["purpose"])


# The overview of this chart:
# 
# Most of the Loans (irrespective of Defaulter or Non-Defaulter) were taken for Debt Consolidation Purposes.
# Least of the Loans (irrespective of Defaulter or Non-Defaulter) were taken for Wedding, Moving, House, Vacation, Educational, Renewable Energy Purposes.
# This shows us that more than half the customers take loan for Debt Consolidation only.

# ## 5. Feature Engineering

# 
# Now, we'll create some new features which can help us in predicting the target (Defaulter or Non-Defaulter).

# In[55]:


'''
Feature Engineering
'''

# Separating the member_id column of test dataframe to help create a csv after predictions
test_member_id = pd.DataFrame(dfTest['member_id'])


# Creating target variable pandas series from train dataframe, this will be used by cross validation to calculate
# the accuracy of the model
train_target = pd.DataFrame(dfTrain['loan_status'])


# It's good to create a copy of train and test dataframes. this way we can play around different features as we tune the
# performance of the classifier with important features
selected_cols = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']
finalTrain = dfTrain[selected_cols]
finalTest = dfTest[selected_cols]

# How big the loan a person has taken with respect to his earnings, annual income to loan amount ratio
finalTrain['loan_to_income'] = finalTrain['annual_inc']/finalTrain['funded_amnt_inv']
finalTest['loan_to_income'] = finalTest['annual_inc']/finalTest['funded_amnt_inv']


# All these attributes indicate that the repayment was not all hunky-dory. All the amounts caclulated are ratios 
# like, recovery to the loan amount. This column gives a magnitude of how much the repayment has gone off course 
# in terms of ratios.
finalTrain['bad_state'] = finalTrain['acc_now_delinq'] + (finalTrain['total_rec_late_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['recoveries']/finalTrain['funded_amnt_inv']) + (finalTrain['collection_recovery_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTrain['funded_amnt_inv'])
finalTest['bad_state'] = finalTest['acc_now_delinq'] + (finalTest['total_rec_late_fee']/finalTest['funded_amnt_inv']) + (finalTest['recoveries']/finalTest['funded_amnt_inv']) + (finalTest['collection_recovery_fee']/finalTest['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTest['funded_amnt_inv'])

# For the sake of this model, I have used just a boolean flag if things had gone bad, with this case I didn't see
# a benifit of including above computations
finalTrain.loc[finalTrain['bad_state'] > 0, 'bad_state'] = 1
finalTest.loc[finalTest['bad_state'] > 0, 'bad_state'] = 1


# Total number of available/unused 'credit lines'
finalTrain['avl_lines'] = finalTrain['total_acc'] - finalTrain['open_acc']
finalTest['avl_lines'] = finalTest['total_acc'] - finalTest['open_acc']


# Interest paid so far
finalTrain['int_paid'] = finalTrain['total_rec_int'] + finalTrain['total_rec_late_fee']
finalTest['int_paid'] = finalTest['total_rec_int'] + finalTest['total_rec_late_fee']



# In[56]:


# Calculating EMIs paid (in terms of percent)
finalTrain['emi_paid_progress_perc'] = ((finalTrain['last_week_pay'].astype(int)/(finalTrain['term'].astype(int)/12*52+1))*100)
finalTest['emi_paid_progress_perc'] = ((finalTest['last_week_pay'].astype(int)/(finalTest['term'].astype(int)/12*52+1))*100)


# In[57]:


# Calculating total repayments received so far, in terms of EMI or recoveries after charge off
finalTrain['total_repayment_progress'] = ((finalTrain['last_week_pay'].astype(int)/(finalTrain['term'].astype(int)/12*52+1))*100) + ((finalTrain['recoveries'].astype(int)/finalTrain['funded_amnt_inv'].astype(int)) * 100)
finalTest['total_repayment_progress'] = ((finalTest['last_week_pay'].astype(int)/(finalTest['term'].astype(int)/12*52+1))*100) + ((finalTest['recoveries'].astype(int)/finalTest['funded_amnt_inv'].astype(int)) * 100)


# In[58]:


finalTrain.replace([np.inf, -np.inf], np.nan, inplace=True)
finalTest.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[59]:


finalTrain.isnull().sum()


# In[60]:


finalTest.isnull().sum()


# In[61]:


finalTrain["bad_state"].fillna(0, inplace = True)
finalTest["bad_state"].fillna(0, inplace = True)
finalTrain["total_repayment_progress"].fillna(0, inplace = True)
finalTest["total_repayment_progress"].fillna(finalTest["total_repayment_progress"].median(), inplace = True)
finalTest["total_repayment_progress"].fillna(finalTest["total_repayment_progress"].median(), inplace = True)

finalTrain.isnull().sum()


# ## 7. Train-Test Split
# We'll spit our data in training and cross-validation sets.

# In[62]:


#Split data set into train-test-cv
#Train model & predict
X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.30)
eval_set=[(X_test, y_test)]


# In[63]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[64]:


print('Initializing xgboost.sklearn.XGBClassifier and starting training...')

st = datetime.now()

clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)

print(datetime.now()-st)

y_pred = clf.predict(X_test)
submission_file_name = 'Submission_'

accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
submission_file_name = submission_file_name + ("_Accuracy_%.6f" % (accuracy * 100)) + '_'

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
submission_file_name = submission_file_name + ("_ROC-AUC_%.6f" % (accuracy_per_roc_auc * 100))

final_pred = pd.DataFrame(clf.predict_proba(np.array(finalTest)))
dfSub = pd.concat([test_member_id, final_pred.ix[:, 1:2]], axis=1)
dfSub.rename(columns={1:'loan_status'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)

import matplotlib.pyplot as plt
print(clf.feature_importances_)
idx = 0
for x in list(finalTrain):
    print('%d %s' % (idx, x))
    idx = idx + 1
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()


# In[ ]:





# In[ ]:




