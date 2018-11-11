import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

#Reading the modified csv file that has been cleand of NaN and
# also added numerical values
df = pd.read_csv("data_profiles.csv")

#easier reading the headers ['age', 'body_type', 'cust_bodyType',
#'diet', 'cust_diet', 'drinks', 'cust_drinks', 'drugs', 'cust_drugs', 'education',
#'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9', 'essay_len',
#'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'orientation', 'cust_orientation', 'pets',
#'religion', 'sex', 'cust_sex', 'sign', 'smokes', 'cust_smokes', 'speaks', 'status', 'cust_status']


# My work evoles around differences between sexes
# 

#First looking at sex and classify that versus the parameters below
# feature_X = df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'height', 'essay_len']]    # Variant 1
# feature_X = df[['cust_smokes', 'cust_drinks', 'cust_drugs']]                           # Variant 2
# feature_X = df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'height']]                 # Variant 3
# feature_X = df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'essay_len']]              # Variant 4
feature_X = df[['height', 'essay_len']]                                                # Variant 5
#feature_X = df[['height']]                                                              # Variant 6

#feature_X = df[['height']]                                                               
#feature_X = df[['cust_smokes']]                                                               
#feature_y = df[['cust_sex']]  

# x = feature_X.values
# min_max_scaler = MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# feature_X = pd.DataFrame(x_scaled, columns=feature_X.columns)

# y = feature_y.values
# y_scaled = min_max_scaler.fit_transform(y)
# feature_y = pd.DataFrame(y_scaled, columns=feature_y.columns)
# feature_Xtr, feature_Xte, feature_ytr, feature_yte = train_test_split(feature_X, feature_y, test_size = 0.2)

# # --------------------------------
# #Trying with LinearRegression
# regression = LinearRegression()
# regression.fit(feature_Xte,feature_yte)

# sex_predict = regression.predict(feature_Xte)
# print "LinearRegression score: " , regression.score(feature_Xtr, feature_ytr) , ". Around 0.42 is not a very good score. 1.0 is the best"

# plt.figure(1)
# plt.xlabel("Height")
# plt.ylabel("Sex")
# plt.title("Sex vs height, LinearRegression")
# plt.plot(feature_Xtr['height'],feature_ytr['cust_sex'],'o', 'r') #red line
# plt.plot(feature_Xte['height'],sex_predict)  #blue
#Using a linear regression here fails misserably, is it due to the duality of the "sex" data?

# --------------------------------
#Trying with KNeighborsClassifier
training_data, validation_data, training_labels, validation_labels = train_test_split (
feature_X, 
feature_y,   #sex
test_size = 0.2, 
random_state = 100)


# iterate and find best k
# With df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'height', 'essay_len']]  # Best score with k= 39 and score is  :  0.8177648040033361 Variant 1 - see score curve
# With df[['cust_smokes', 'cust_drinks', 'cust_drugs']]                         # Best score with k= 39  and score is :  0.5939115929941619 Variant 2
# With df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'height']]               # Best score with k= 8   and score is :  0.8212677231025854 Variant 3
# With df[['cust_smokes', 'cust_drinks', 'cust_drugs', 'essay_len']]            # Best score with k= 184 and score is :  0.5905754795663053 Variant 4
# With df[['height', 'essay_len']]                                              # Best score with k= 94  and score is :  0.8254378648874062 Variant 5
# With df[['height']]                                                           # Best score with k= 131 and score is :  0.8247706422018348 Variant 6
# 
best_score = 0
best_k = 0
accuracies = []
k_list = range(1,200) # Tested with 1-200 and saved png, but has reduced the range after first tun to reduce compute xtime
for i in k_list:    
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels.values.ravel())
  print"i: " ,i, "score : ",classifier.score(validation_data, validation_labels)
  accuracies.append(classifier.score(validation_data, validation_labels.values.ravel()))
  if classifier.score(validation_data, validation_labels)>best_score:
    best_score = classifier.score(validation_data, validation_labels)
    best_k = i
print "Best score with k=", best_k, "and score is : ", best_score

plt.figure(2)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Sex Classifier Accuracy - Variant 6")


classifier = KNeighborsClassifier(n_neighbors = best_k)
classifier.fit(training_data, training_labels.values.ravel())
print "KNeighborsClassifier score: ", classifier.score(validation_data, validation_labels) , " is way better than from the LinearRegression"



plt.show()

#print feature_data











