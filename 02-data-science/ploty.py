from pathlib import Path
import pandas as pd
import plotly.express as px

folder = Path(__file__).parent
csv_file = folder / "tourism_custom_data.csv"

df = pd.read_csv(csv_file)

#fig = px.line( df, x="Month", y="Tourist_Arrivals",   title="Monthly Tourist Arrivals")

# bar chart 
#fig = px.bar(
   # df,    x= "Country", y="Tourist_Arrivals",    title = "Tourist Arrivals by country"   )

#scattrer plot

#fig = px.scatter(df, x='Average_Travel_Cost',y='Hotel_Bookings',title= 'Hotel Bookings by Country')

#viloin plot
#fig = px.violin(df,x='Season ',y='Average_Travel_Cost',title='Seasonal Variation in Travel Cost')

#gantt chart 
#fig = px.timeline(df,x_start='Start_Date',x_end='End_Date',y='Activity',title='Tourism Itenerary')

#3d plot

fig = px.scatter_3d(df,x='Average_Travel_Cost',y='Hotel_Bookings',z='Hotel_Bookings',
                    title='3D Analysis of Travel Cost,Rating & Bookings', color='Hotel_Rating')

fig.show()