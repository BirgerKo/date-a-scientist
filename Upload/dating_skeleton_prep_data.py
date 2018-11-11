import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
# print(sys.version) #Definately 2.7.10

#Create the df:
df = pd.read_csv("profiles.csv")
#print df.shape  # (59946, 31)
#nan_rows = df[df['height'].isnull()]
#print "nan_rows : ", nan_rows,"\n"
#print df.loc[36428,:]
print ("List df", list(df)) 
print "\n"


# Defining functions for cleaning NaN
# It evolved into this
def replace_nan():
    headers = (list(df))
    for i in range(len(headers)):
        if df.dtypes[i] == "O": #Checks if the column is of type object
            #print ("O: ",headers[i])
            if "essay" in headers[i]:
                df[headers[i]].fillna("", inplace=True)  # In essay fileds, just remove it, no explanation
            else:
                df[headers[i]].fillna("No answer", inplace=True) #If so remove all NaN and replace with No answer text
        elif df.dtypes[i] == "int64": #Checks if the column is of type int64
            #print ("int64: ",headers[i])
            header = headers[i]
            #df[header] = df[header][np.logical_not(np.isnan(df[header]))] #Takes out the NaN of numbers
            df[headers[i]].fillna(0, inplace=True)
        elif df.dtypes[i] == "float64": #Checks if the column is of type int64
            df[headers[i]].fillna(0.0, inplace=True)  #Replaceing float numbers with 0.0
        else:
            print ("Unknown datatype detected: ", df.dtypes[i])
            return 1 #Failure
            
    return 0


def make_mapping(header):               #Need to run per header I want to examin
    #print df[header].unique()
    map_labels = df[header].unique()
    map_dict = {}
    for i in range(len(map_labels)):
        map_dict[map_labels[i]] = i
    return map_dict
#Start doing stuff
print ("Cleaning up NaN and mapping string data to numbers")
	
#Map "body_type" to numbers
df["cust_bodyType"] = df.body_type.map(make_mapping("body_type"))

#Map "diet" to numbers
df["cust_diet"] = df.diet.map(make_mapping("diet"))

#Drinks mapping
df["cust_drinks"] = df.drinks.map(make_mapping("drinks"))

#Drugs mapping
df["cust_drugs"] = df.drugs.map(make_mapping("drugs"))

#Education mapping
df["cust_education"] = df.education.map(make_mapping("education"))

#Sexual orinetation mapping
df["cust_orientation"] = df.orientation.map(make_mapping("orientation"))

#Smoking  mapping
df["cust_smokes"] = df.smokes.map(make_mapping("smokes"))

#Status mapping
df["cust_status"] = df.status.map(make_mapping("status"))

#Sex mapping
df["cust_sex"] = df.sex.map(make_mapping("sex"))


#Take out all NaN in the dataset
replace_nan()

#Essay prep
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Combining the essays
# print (df[essay_cols])
all_essays = df[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))


#Re-arrange the columns, more logical for me
df=df[['age', 'body_type','cust_bodyType', 'diet', 'cust_diet', 'drinks', "cust_drinks", 'drugs', 'cust_drugs', 
'education','essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9', "essay_len", 
'ethnicity', 'height', 'income', 'job', 'last_online', 'location', 'offspring', 'orientation', "cust_orientation", 
'pets', 'religion', 'sex', 'cust_sex', 'sign', 'smokes', 'cust_smokes','speaks', 'status', 'cust_status']]

#Write the updated file so I do not have to do this every time
#Reuse this file in the dating_skeleton.py file
df.to_csv("data_profiles.csv", index=False) #When reading this CSV again a new column "Unnamed: 0" is added if _index=False is not there
print ("List df", list(df)) 
#print (df.head())
#print (df.loc[36428,:])