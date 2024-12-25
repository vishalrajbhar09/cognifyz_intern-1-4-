#!/usr/bin/env python
# coding: utf-8

# ### VISHAL RAMKUMAR RAJBHAR
# ### vishalrajbhar.0913@gmail.com
# 

# In[11]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
file_path = r'C:\Users\vishal rajbhar\Downloads\Dataset .csv'  # Update the file path
data = pd.read_csv(file_path)

# Step 2: Handle missing values
# Fill missing values in the 'Cuisines' column with 'Unknown'
data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# Step 3: Encode the target variable (Cuisines)
# Convert cuisine names into numerical labels
label_encoder = LabelEncoder()
data['Cuisine_Label'] = label_encoder.fit_transform(data['Cuisines'])

# Step 4: Select features (X) and target variable (y)
# Using selected features for simplicity
X = data[['Average Cost for two', 'Price range', 'Votes']]  # Add more features if needed
y = data['Cuisine_Label']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a classification model
# Using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
# Make predictions on the test set
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
# Use only the labels found in `y_test` to avoid mismatch errors
print("\nClassification Report:")
labels_in_test = sorted(y_test.unique())  # Unique classes in the test set
print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=label_encoder.inverse_transform(labels_in_test)))

# Step 8: Analyze results
# Display the most important features
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances)

