import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


# scatter plots.

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
plt.show()

