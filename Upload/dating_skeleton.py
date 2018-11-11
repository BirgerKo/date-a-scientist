import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

#Create your df here:
#Reading the modified csv file that has been cleand of NaN and
# also added numerical values
df = pd.read_csv("data_profiles.csv")

#print (df.age.head())
#print (df.sex.head())
#print (df.shape) # Quite large database! 59946 rows and 31 columns
#print (list(df)) 
#easier reading the headers ['age', 'body_type', 'cust_bodyType',
#'diet', 'cust_diet', 'drinks', 'cust_drinks', 'drugs', 'cust_drugs', 'education',
#'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9', 'essay_len',
#'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'orientation', 'cust_orientation', 'pets',
#'religion', 'sex', 'sign', 'smokes', 'cust_smokes', 'speaks', 'status', 'cust_status']

# print (df["status"].unique())
# print (df.income)

plt.figure(1)
plt.subplot(3,2,1)
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.subplot(3,2,2)
plt.scatter(df.age, df.height)
plt.xlabel("Age")
plt.ylabel("height")
''' Remove outliers. Age should be limited from age 20 - 80. Height below 40 inches and above 85 is unrealistic 
core groupo seems to be between 20 and 40 yearsa of age'''
''' Dropping bad rows
First dropping age under 20'''
df_clean = df[df['age'] > 20]
'''dropping above 80'''
df_clean = df_clean[df_clean['age'] < 80]
''' dropping hight below 40 and above 85 inches'''
df_clean = df_clean[df_clean['height'] > 40]
df_clean = df_clean[df_clean['height'] < 85]

df_values = df_clean[['age','cust_bodyType', 'cust_diet', 'cust_drinks', 'cust_drugs', 'essay_len', 'height', 'income', 'cust_orientation', 'cust_smokes', 'cust_status']]

plt.subplot(3,2,3)
#plt.xlim(20,80)
plt.hist(df_clean.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.subplot(3,2,4)
#plt.xlim(20,80)
#plt.ylim(40,85)
plt.scatter(df_clean.age, df_clean.height)
plt.xlabel("Age")
plt.ylabel("Height")

plt.subplot(3,2,6)
robust_scaler = RobustScaler()
Xtr_r = robust_scaler.fit_transform(df_values)
plt.scatter(Xtr_r[:, 0], Xtr_r[:, 1])
plt.xlabel("Age")
plt.ylabel("Body Type")


# plt.subplot(3,2,6)
# #plt.xlim(20,80)
# Xx = RobustScaler(quantile_range=(25, 75)).fit_transform([df_clean.age, df_clean.age])
# plt.hist(Xx, bins=20)
# plt.xlabel("Age")
# plt.ylabel("Frequency")


# plt.subplot(3,2,2)
# plt.hist(df.height, bins=20)
# plt.xlabel("Height")
# plt.ylabel("Frequency")
# plt.xlim(40,95)
# plt.subplots_adjust(bottom=0.15)

# plt.subplot(3,2,3)
# plt.hist(df.cust_status, bins = 5)
# plt.xlabel("Status")
# labels=df["status"].unique()
# plt.xticks(range(0,len(labels)),labels, rotation = 75)
# plt.ylabel("Frequency")

# plt.subplot(3,2,4)
# plt.hist(df.cust_smokes, bins = 6)
# plt.xlabel("Smokes")
# labels=df["smokes"].unique()
# plt.xticks(range(0,len(labels)),labels, rotation = 75)
# plt.ylabel("Frequency")
# plt.subplots_adjust(bottom=0.15)

# plt.show()
# plt.close()


# plt.subplot(2,2,1)
# plt.xlabel("Age")
# plt.ylabel("Smokes")
# plt.scatter(df.age, df.cust_smokes)
# plt.subplot(2,2,2)
# plt.xlabel("Income")
# plt.ylabel("Drugs")
# plt.xlim(0, 200000) # Taking out the extrems limiting to 200k usd income
# plt.scatter(df.income, df.cust_drugs)
#Very evident that there are outliers that should be looked at.
# plt.subplot(2,2,3)
# plt.xlabel("Age")
# plt.ylabel("Income")
#plt.xlim(20, 80) # Taking out the extrems limiting to 200k usd income
# plt.ylim(10, 200000) # Taking out the extrems limiting to 200k usd income
# plt.scatter(df.age, df.income)
# plt.subplot(2,2,4)
# plt.xlabel("Body Type")
# plt.ylabel("Diet")
#plt.xlim(20, 80) # Taking out the extrems limiting to 200k usd income
#plt.ylim(10, 200000) # Taking out the extrems limiting to 200k usd income
# plt.scatter(df.cust_bodyType, df.cust_smokes)

# plt.show()
# plt.close()

'''linear regression
# easier reading the headers ['age', 'body_type', 'cust_bodyType',
# 'diet', 'cust_diet', 'drinks', 'cust_drinks', 'drugs', 'cust_drugs', 'education',
# 'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9', 'essay_len',
# 'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'orientation', 'cust_orientation', 'pets',
# 'religion', 'sex', 'sign', 'smokes', 'cust_smokes', 'speaks', 'status', 'cust_status']'''

# x = df[['cust_diet','cust_drinks', 'cust_drugs', 'cust_orientation', 'cust_smokes', 'cust_status', 'age', 'height']]
# y = df[['cust_bodyType']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# lm = LinearRegression()
# model=lm.fit(x_train, y_train)
# y_predict = lm.predict(x_test)

##plot the linear regression
# plt.scatter(y_test, y_predict, alpha=0.2)
# plt.xlabel("Body type")
# plt.ylabel("Predicted Body type")
# labels=df["body_type"].unique()
# plt.xticks(range(0,len(labels)),labels, rotation = 75)
# plt.show()

# print(lm.score(x_train,y_train)) # Horrible values 0.017977705759686047
# print(lm.score(x_test,y_test))   # Horrible values 0.018112844639043613

# residuals = y_predict - y_test                #Do not quite get this one yet....
# plt.scatter(y_predict, residuals, alpha=0.4)
# plt.title('Residual Analysis')
# plt.show()

##next try
x = df[['cust_diet','cust_drinks', 'cust_drugs', 'cust_smokes', 'cust_status', 'age', 'height']]
y = df[['cust_bodyType']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()
model=lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

'''plot the linear regression'''
plt.figure(2)
plt.subplot(2,1,1)
plt.scatter(y_test, y_predict, alpha=0.2)
plt.xlabel("Body type")
plt.ylabel("Predicted Body type")
labels=df["body_type"].unique()
plt.xticks(range(0,len(labels)),labels, rotation = 75)
plt.title("Without orientation")


print(lm.score(x_train,y_train)) # Horrible values 0.017977705759686047
print(lm.score(x_test,y_test))   # Horrible values 0.018112844639043613

# residuals = y_predict - y_test                #Do not quite get this one yet....
# plt.scatter(y_predict, residuals, alpha=0.4)
# plt.title('Residual Analysis')
# plt.show()

##next try with RobustScaler performed on the values
x = df_values[['cust_diet','cust_drinks', 'cust_drugs', 'cust_smokes', 'cust_status', 'age', 'height']]
y = df_values[['cust_bodyType']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()
model=lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

'''plot the linear regression'''
plt.subplot(2,1,2)
plt.scatter(y_test, y_predict, alpha=0.2)
plt.xlabel("Body type")
plt.ylabel("Predicted Body type")
labels=df["body_type"].unique()
plt.xticks(range(0,len(labels)),labels, rotation = 75)
plt.title("Without orientation, RobustScaler")
#plt.show()

print(lm.score(x_train,y_train)) # Still horrible values

# residuals = y_predict - y_test                #Do not quite get this one yet....
# plt.scatter(y_predict, residuals, alpha=0.4)
# plt.title('Residual Analysis')

''' Age and income'''
plt.figure(3)
age_income = df.loc[:,['age', 'income']]

age_income = age_income[(age_income[['income']] != -1).all(axis=1)]

X_train, X_test = train_test_split(age_income, test_size = 0.2)

Xtr = X_train[['age']]
robust_scaler = RobustScaler()
Xtr_r = robust_scaler.fit_transform(Xtr)

ytr = X_train[['income']]
robust_scaler = RobustScaler()
ytr_r = robust_scaler.fit_transform(ytr)

regression = LinearRegression()
regression.fit(Xtr_r,ytr_r)

y_predicted = regression.predict(Xtr_r)

X = X_test[['age']]
y = X_test[['income']]

print('Age v. Income Regression Score with RobustScaling: ', regression.score(X,y))
plt.title("Age vs income with use of \"RobustScaler\"")
plt.xlabel('Age')
plt.ylabel('Income')
plt.scatter(X,y,alpha=0.2)
plt.plot(Xtr_r, y_predicted)
plt.xlim(20,80)
plt.ylim(20000,200000)



''' Age and income without RobustScaler'''
plt.figure(4)
age_income = df.loc[:,['age', 'income']]

age_income = age_income[(age_income[['income']] != -1).all(axis=1)]

X_train, X_test = train_test_split(age_income, test_size = 0.2)

Xtr = X_train[['age']]
ytr = X_train[['income']]

regression = LinearRegression()
regression.fit(Xtr,ytr)

y_predicted = regression.predict(Xtr)

X = X_test[['age']]
y = X_test[['income']]

print('Age v. Income Regression Score: ', regression.score(X,y))
plt.title("Age vs income WITHOUT use of \"RobustScaler\"")
plt.xlabel('Age')
plt.ylabel('Income')
plt.scatter(X,y,alpha=0.2)
plt.plot(Xtr, y_predicted)
plt.xlim(20,80)
plt.ylim(20000,200000)
plt.show()
plt.close()










