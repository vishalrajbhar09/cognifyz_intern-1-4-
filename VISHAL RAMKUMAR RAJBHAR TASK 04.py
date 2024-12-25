#!/usr/bin/env python
# coding: utf-8

# ### VISHAL RAKUMAR RAJBHAR
# ### vishalrajbhar.0913 @gmail.com

# In[3]:


pip install folium


# In[6]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium





# Step 1: Load the dataset
file_path = r'C:\Users\vishal rajbhar\Downloads\Dataset .csv'  
data = pd.read_csv(file_path)





# Step 2: Handle missing values
data['Cuisines'] = data['Cuisines'].fillna('Unknown')





# Step 3: Basic Exploration of Latitude and Longitude
if 'Latitude' in data.columns and 'Longitude' in data.columns:
    print("Latitude and Longitude Stats:")
    print(data[['Latitude', 'Longitude']].describe())
else:
    print("Latitude and Longitude columns are missing.")


    
    
    
# Step 4: Plot the locations of restaurants on a map
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=10)

for idx, row in data.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Restaurant Name']
    ).add_to(restaurant_map)

restaurant_map.save("simple_restaurant_map.html")
print("Restaurant map saved as 'simple_restaurant_map.html'.")





# Step 5: Grouping by City
city_group = data.groupby('City').agg({
    'Restaurant ID': 'count',
    'Aggregate rating': 'mean'
}).reset_index()
city_group.rename(columns={
    'Restaurant ID': 'Restaurant Count',
    'Aggregate rating': 'Average Rating'
}, inplace=True)





# Display top 5 cities by restaurant count
print("\nTop 5 Cities by Restaurant Count:")
print(city_group.sort_values(by='Restaurant Count', ascending=False).head())





# Step 6: Visualize Top Cities by Restaurant Count
top_cities = city_group.sort_values(by='Restaurant Count', ascending=False).head(15)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_cities, x='City', y='Restaurant Count', palette='viridis')
plt.title('Top 10 Cities by Restaurant Count')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()





# Step 7: Grouping by Locality and Analyzing Ratings
locality_group = data.groupby('Locality').agg({
    'Restaurant ID': 'count',
    'Aggregate rating': 'mean'
}).reset_index()
locality_group.rename(columns={
    'Restaurant ID': 'Restaurant Count',
    'Aggregate rating': 'Average Rating'
}, inplace=True)




# Display top 5 localities by average rating
print("\nTop 5 Localities by Average Rating:")
print(locality_group.sort_values(by='Average Rating', ascending=False).head())

# Step 8: Heatmap of Restaurant Concentrations
if 'Latitude' in data.columns and 'Longitude' in data.columns:
    heatmap_map = folium.Map(location=map_center, zoom_start=15)
    heatmap_data = data[['Latitude', 'Longitude']].dropna().values.tolist()

    from folium.plugins import HeatMap
    HeatMap(heatmap_data, radius=8).add_to(heatmap_map)

    heatmap_map.save("simple_restaurant_heatmap.html")
    print("Heatmap saved as 'simple_restaurant_heatmap.html'.")
else:
    print("Cannot create a heatmap due to missing latitude and longitude.")

# Insights
print("\nInsights:")
print("- Cities with the highest number of restaurants tend to have more diverse cuisines.")
print("- Localities with high ratings suggest popular or high-quality restaurants.")
print("- Heatmap reveals the clusters of restaurants, often in urban centers.")

