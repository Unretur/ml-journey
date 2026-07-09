import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# sctter plots 

df = pd.read_csv("telecom_customer_data.csv")
df.head()
df.shape


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Call_Duration",y="Satisfaction_Score",hue="Network_Provider")

#move teh legend outside the plot 
plt.legend(title ="Network Provider",bbox_to_anchor=(1,1),loc="upper left")

#add labels and tittle
plt.xlabel("Call Duration (minutes)")
plt.ylabel("Satisfaction Score")
plt.title("Call Duration vs. Satisfaction Score")

#show  the plot
#plt.show()

# this is categorical plot 


plt.figure(figsize=(6,4))
sns.countplot(data=df,x= 'Network_Provider',hue='Network_Provider',palette ='Set2')
plt.title("Customer Distribution by Network Provider")
plt.xlabel("Network Provider")
plt.ylabel(" Count of Customer")
plt.legend().remove()  
# plt.show()

#MATRRIX PLOT
plt.figure(9.6)
numeric_df = df.select_dtypes(include=['number']) # slect only numeric forms 
sns.heatmap(numeric_df.corr(),annot= True, cmap='coolwarm',fmt="2f",linewidths=0.5)

plt.title("correlation matrix of features")
plt.show()