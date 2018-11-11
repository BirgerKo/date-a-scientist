import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

#Createing the df :
#Reading the modified csv file that has been cleand of NaN and
# also added numerical values
df = pd.read_csv("data_profiles.csv")

#exploring fields
print ("List df", list(df))
''' ['age', 'body_type', 'cust_bodyType', 'diet', 'cust_diet', 'drinks', 'cust_drinks', 'drugs', 'cust_drugs', 'education', 
'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9', 'essay_len', 'ethnicity', 
'height', 'income', 'job', 'last_online', 'location', 'offspring', 'orientation', 'cust_orientation', 'pets', 'religion',
 'sex', 'cust_sex', 'sign', 'smokes', 'cust_smokes', 'speaks', 'status', 'cust_status']'''
 
print (df['body_type'].unique())
'''Values:
'a little extra' 'average' 'thin' 'athletic' 'fit' 'No answer' 'skinny'
 'curvy' 'full figured' 'jacked' 'rather not say' 'used up' 'overweight' ''' 

print (df['diet'].unique())
'''Values:
'strictly anything' 'mostly other' 'anything' 'vegetarian' 'No answer'
 'mostly anything' 'mostly vegetarian' 'strictly vegan'
 'strictly vegetarian' 'mostly vegan' 'strictly other' 'mostly halal'
 'other' 'vegan' 'mostly kosher' 'strictly halal' 'halal'
 'strictly kosher' 'kosher' ''' 
 
print (df['drinks'].unique())
'''Values:
'socially' 'often' 'not at all' 'rarely' 'No answer' 'very often'
 'desperately' '''  

print (df['drugs'].unique())
'''Values:
'never' 'sometimes' 'No answer' 'often' '''  

print (df['education'].unique())
'''Values seem to be free text:
'working on college/university' 'working on space camp'
 'graduated from masters program' 'graduated from college/university'
 'working on two-year college' 'No answer' 'graduated from high school'
 'working on masters program' 'graduated from space camp'
 'college/university' 'dropped out of space camp'
 'graduated from ph.d program' 'graduated from law school'
 'working on ph.d program' 'two-year college'
 'graduated from two-year college' 'working on med school'
 'dropped out of college/university' 'space camp'
 'graduated from med school' 'dropped out of high school'
 'working on high school' 'masters program' 'dropped out of ph.d program'
 'dropped out of two-year college' 'dropped out of med school'
 'high school' 'working on law school' 'law school'
 'dropped out of masters program' 'ph.d program'
 'dropped out of law school' 'med school' '''  

print (df['ethnicity'].unique())
''' List of waht the customer prefers in a partner, multiple
ex:  asian, middle eastern, black, native american, indian, pacific islander, hispanic / latin, white, other'''  

print (df['job'].unique())
''' ['transportation' 'hospitality / travel' 'No answer' 'student'
 'artistic / musical / writer' 'computer / hardware / software'
 'banking / financial / real estate' 'entertainment / media'
 'sales / marketing / biz dev' 'other' 'medicine / health'
 'science / tech / engineering' 'executive / management'
 'education / academia' 'clerical / administrative'
 'construction / craftsmanship' 'rather not say' 'political / government'
 'law / legal services' 'unemployed' 'military' 'retired']'''

print (df['last_online'].head())
''' Dates - format:  2012-06-28-20-30 , year, month, day, hour, minutes'''  

print (df['location'].unique())
'''  Massive list that seems to be partly state, partly country and then also city
'vancouver, british columbia, canada' 'muir beach, california'
 'pacheco, california' 'irvine, california' 'kansas city, missouri'
 'kassel, germany' 'canyon, california' 'philadelphia, pennsylvania'
 'oceanview, california' 'long beach, new york' 'amsterdam, netherlands'
 'taunton, massachusetts' 'napa, california' 'austin, texas'''  

print (df['offspring'].unique())
#Maybe look at making a numerical based on has, has&more, or doesn't&want&kid and doesn't&want etc.?
'''Values:
 'doesn&rsquo;t have kids, but might want them' 'No answer'
 'doesn&rsquo;t want kids' 'doesn&rsquo;t have kids, but wants them'
 'doesn&rsquo;t have kids' 'wants kids' 'has a kid' 'has kids'
 'doesn&rsquo;t have kids, and doesn&rsquo;t want any'
 'has kids, but doesn&rsquo;t want more'
 'has a kid, but doesn&rsquo;t want more' 'has a kid, and wants more'
 'has kids, and might want more' 'might want kids'
 'has a kid, and might want more' 'has kids, and wants more']'''  
 
print (df['orientation'].unique())
'''Values:
 'straight' 'bisexual' 'gay' '''  
 
print (df['pets'].unique())
# Numerical with dogs, cats in combionation with dislike?
'''Values:
 'likes dogs and likes cats' 'has cats' 'likes cats' 'No answer'
 'has dogs and likes cats' 'likes dogs and has cats'
 'likes dogs and dislikes cats' 'has dogs' 'has dogs and dislikes cats'
 'likes dogs' 'has dogs and has cats' 'dislikes dogs and has cats'
 'dislikes dogs and dislikes cats' 'dislikes cats'
 'dislikes dogs and likes cats' 'dislikes dogs' '''   
 
print (df['sex'].unique())
'''Values:
'm' 'f' '''  
 
print (df['sign'].unique())
'''Values: lots....
 'capricorn and it&rsquo;s fun to think about' 'leo'
 'aries but it doesn&rsquo;t matter' 'aries'
 'scorpio but it doesn&rsquo;t matter'
 'sagittarius and it&rsquo;s fun to think about'
 'libra and it matters a lot' 'taurus and it&rsquo;s fun to think about'
 'leo and it matters a lot' 'virgo and it&rsquo;s fun to think about'
 'cancer and it matters a lot' 'capricorn' 'pisces and it matters a lot'
 'aries and it matters a lot' 'capricorn and it matters a lot'
 'aquarius and it matters a lot' 'sagittarius and it matters a lot' '''  

print (df['smokes'].unique())
'''Values:
'sometimes' 'no' 'No answer' 'when drinking' 'yes' 'trying to quit' '''  
 
print (df['speaks'].unique())
#Maybe add a count of languages to the df?
'''Values, probably lots of variants. 
 'english, french, c++' ...
 'english (fluently), hindi (poorly), french (poorly), tamil (okay), spanish (poorly)' '''  
 
print (df['status'].unique())
'''Values:
'single' 'available' 'seeing someone' 'married' 'unknown' '''  

#print (df.job.unique())


