import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot  as plt


#set theme 
sns.set(style="whitegrid")
plt.rcParams['figure.figsize']=(12,9)

df=pd.read_csv("car_details_from_car_dekho.csv")
df.head(5)
df.shape
#print(df.shape)

#fueltype responsibility 
#price range anlayiszis

sns.histplot(df['selling_price'],kde=True,bins=30,color='pink')
plt.title('selling price distribution')
plt.xlabel('Price(INR)')
# plt.show()

#  boxplot
sns.boxplot(x='transmission',hue='transmission',y='selling_price',data=df,palette='coolwarm')
plt.title('Selling Price by Transmission Type')
plt.show()
