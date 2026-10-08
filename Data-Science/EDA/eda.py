from pathlib import Path
from numpy._core import numeric
import pandas as pd

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "data_clean.csv") #reading csv file

df.head(5)

data_bkup = df  # taking back up before any changes , so we can restore the data 

df.tail(5)
#print(df.tail(5))

#  df.info()    # its get us info on coloums , null sets is there and what isn the type 
#print(df.info())

df.shape
#print(df.shape)

df.dtypes
#print(df.dtypes)

df.describe()
#print(df.describe())

df['Month'].head(25)
#print(df['Month'].head(25))

df['Month']=df['Month'].replace('May',5)

df['Month']=pd.to_numeric(df['Month'],errors='coerce')   # errors ='corece' force to convert any other value like may into na
df['Temp C']=pd.to_numeric(df['Temp C'],errors='coerce')
df['Weather']=(df['Weather'].astype('category'))

df.info()
#print(df.info())

df.duplicated()  # help us to find duplicates in terms false - no , true - its a duplicated
#print(df.duplicated())

#count of duplicates rows
df[df.duplicated()].shape
#print(df[df.duplicated()].shape)

df[df.duplicated()]    # it will print duplicated data drames
#print(df[df.duplicated()])

df=df.drop_duplicates()    # it will droop duplicates from data sets 
#print(df.drop_duplicates())

# DROPS COLUMS WHICH ARE UNNESSARY 

data_bkup1=df  #it will have backup

df = df.drop('Unnamed: 0', axis=1)  # it will drop the coloumn 

df.head()
#print(df.head())

df = df.drop('Temp C', axis=1)

df.head()
#print(df.head())

df= df.rename({'Solar.R':'Solar'},axis=1) # it will rename the coloumn

df.head()
#print(df.head())

cols= df.columns #its print  all the coloumns are there

#print(cols)

import seaborn as sns
cols = df.columns
#colours =['000099','#ffff00']# specific the colours -  yellow  iis missing . blue is not 
#sns.heatmap(df[cols].isnull(),
           # cmap=sns.color_palette(colours)


df.isnull().sum()
print(df.isnull().sum())

#calculate q1 and q3 . to chech there is outlier or not 
Q1 = df["Ozone"].quantile(0.25)
Q3 = df["Ozone"].quantile(0.75)

IQR = Q3 - Q1


#define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# find outlier 
outlier = df[(df["Ozone"] < lower_bound) | (df["Ozone"] > upper_bound)]

#print("Number of outliers in Wind column :", outlier.shape[0])

ozone_median = df["Ozone"].median()

df["Ozone"].fillna(ozone_median, inplace=True)

#print("Filled with Median :",ozone_median)

#print("Missing in Ozone :",df['Ozone'].isnull().sum())

#replace na with mean ,a s no outlier column
# calculate  mean of the " solar "coloumn (excluding na )

mean_value = df["Solar"].mean()
mean_value
#print(mean_value)

#replacing na value witrh mean 
df["Solar"].fillna(mean_value, inplace=True)

#aafter filling 
#print("Missing values after:",df['Solar'].isna().sum())

#check ost frequent value (mode)
# mode is used in object not in numerical cladd
mode_value = df['Weather'].mode()[0]
mode_value
print(mode_value)

#replace na with mode 
df['Weather'].fillna(mode_value,inplace=True)

df.isnull().sum()
print(df.isnull().sum())


df=pd.get_dummies(df,columns=['Weather'])
#print(df)

# normalization of the data 
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler


df.values 
#print(df.values)

array = df.values
array
print(array)