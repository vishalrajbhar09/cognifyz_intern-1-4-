#!/usr/bin/env python
# coding: utf-8

# ### VISHAL RAMKUMAR RAJBHAR
# ### vishalrajbhar.0913@gmail.com

# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r'C:\Users\vishal rajbhar\Downloads\Dataset .csv'  # Adjust the file path
data = pd.read_csv(file_path)

# Display dataset info and check for missing values
print("Dataset Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 2: Identify features (X) and target (y)
target_column = 'Aggregate rating'  # Adjust the column name if necessary
X = data.drop(columns=[target_column])  # Drop the target column
y = data[target_column]  # Target variable

# Step 3: Identify numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Step 4: Preprocessing - Handle missing values and encode categorical variables
numerical_transformer = SimpleImputer(strategy='mean')  # Fill missing numerical data with mean
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical data with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convert categories to one-hot encoding
])

# Combine preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 6: Define and train the regression model
model = RandomForestRegressor(random_state=42)  # Use Random Forest for regression

# Create a pipeline that includes preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train the model
pipeline.fit(X_train, y_train)
print("\nModel training completed.")

# Step 7: Evaluate the model
# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 8: Analyze Feature Importance (if supported by the model)
if hasattr(model, 'feature_importances_'):
    # Get feature names from the preprocessor
    onehot_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(cat_features)
    all_feature_names = list(num_features) + list(onehot_feature_names)

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance_df.head(10))  # Display top 10 influential features

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

# Step 9: Save the pipeline for future use (optional)
import joblib
joblib.dump(pipeline, 'restaurant_rating_predictor.pkl')
print("\nModel pipeline saved as 'restaurant_rating_predictor.pkl'.")

