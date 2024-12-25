#!/usr/bin/env python
# coding: utf-8

# ### VISHAL RAMKUMAR RAJBHAR
# ### vishalrajbhar.0913@gmail.com

# In[4]:


# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the dataset
file_path = r'C:\Users\vishal rajbhar\Downloads\Dataset .csv'  
data = pd.read_csv(file_path)





# Display the first few rows to understand the dataset
print("Dataset Preview:")
print(data.head())



# Step 2: Preprocess the dataset
# Handle missing values
data['Cuisines'] = data['Cuisines'].fillna('Unknown')  # Replace missing cuisines with 'Unknown'




# Combine the  relevant columns for recommendation purposes.
data['Combined_Features'] = data['Cuisines'] + ' ' + data['Price range'].astype(str)

# Display the combined features column
print("\nCombined Features for Recommendations:")
print(data[['Restaurant Name', 'Combined_Features']].head())

# Step 3: Encode categorical variables using TF-IDF
#The TfidfVectorizer is used to encode text data (cuisines and price range) into numerical vectors suitablefor similarity computation.
# Use TfidfVectorizer to convert text features into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
feature_matrix = tfidf.fit_transform(data['Combined_Features'])

print("\nTF-IDF Matrix Shape:", feature_matrix.shape)

# Step 4: Define the recommendation system
def recommend_restaurants(user_preferences, top_n=10):
    """
    Recommends restaurants based on user preferences using cosine similarity.

    Args:
        user_preferences (str): User's input preferences (e.g., preferred cuisines, price range).
        top_n (int): Number of top recommendations to return.

    Returns:
        recommendations (list): List of recommended restaurant names.
    """
    # Transform user preferences into the same vector space as the TF-IDF matrix
    user_vector = tfidf.transform([user_preferences])
    
    # Compute cosine similarity between user preferences and all restaurants
    cosine_sim = cosine_similarity(user_vector, feature_matrix)
    
    # Get the indices of top matching restaurants
    top_indices = np.argsort(cosine_sim[0])[::-1][:top_n]
    
    # Retrieve and return the names of the recommended restaurants
    recommendations = data.iloc[top_indices]['Restaurant Name'].values
    return recommendations



# Step 5: Test the recommendation system
# Define sample user preferences
sample_preferences = "Italian 2" 


# Get recommendations
print("\nUser Preferences:", sample_preferences)
recommended_restaurants = recommend_restaurants(sample_preferences, top_n=10)





# Display recommendations
print("\nRecommended Restaurants from the list of file:")
for i, restaurant in enumerate(recommended_restaurants, 1):
    print(f"{i}. {restaurant}")

    


