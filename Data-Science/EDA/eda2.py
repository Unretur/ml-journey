from pathlib import Path
from numpy._core import numeric
import pandas as pd

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "insurance.csv") #reading csv file

df.head(5)
#print(df.head(5))

#Data Transformationn

import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

sns.histplot(df['charges'],kde =True) # bfore transformation

#box-cos on postove bmi 
from scipy.stats import boxcox

df['bmi_boxcoc'],_=boxcox(df['bmi']+1)
#print(df)

df['log_charges']= np.log1p(df['charges'])

#print(df)

sns.histplot(df['charges'],kde =True)


#whenever histogram show data distribution in a bell shaped curve
# it is normal distribution , if not its skewed distribution 

## feature enegineering ... create  meaningfull features 

df.head()   # BEFIRE APPLYING FEATURE ENEGINEERING 
#print(df.head())

# craeting cateogry as per bmi (body mass index)and age 

df['bmi_cat']= pd.cut(df['bmi'],bins=[0,18.5,25,30,100],labels=['under','normal','over','obese'])

df['age_group']= pd.cut(df['age'],bins= [17,30,45,60,100],labels=['young','adult','mature','senior'])

df['age_smoker'] = df['age']*(df['smoker']=='yes')

# AFTER APPLYING FEATURE ENEGINEERING 
df.head()
#print(df.head())

for col in ['sex','smoker','bmi_cat','age_group']:
    print(col, df[col].value_counts() )
   # print()

#Target variable disstribution 

