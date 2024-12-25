# cognifyz_intern-task one



Restaurant Rating Prediction using Random Forest

This Python program predicts the aggregate rating of restaurants based on a dataset of features. It leverages preprocessing, feature engineering, and a Random Forest Regressor model to achieve accurate predictions. 
Below Aare the breakdown of the program:

Steps in the Program
Importing Libraries:
Essential libraries such as pandas, numpy, scikit-learn, and matplotlib are imported for data processing, model training, and visualization.

Loading the Dataset:
The dataset is loaded from a CSV file into a DataFrame. Basic information about the dataset is displayed, including the number of rows, columns, and missing values.

Identifying Features and Target:

The target variable (y) is identified as the "Aggregate rating" column.
The feature set (X) is derived by dropping the target column from the dataset.
Identifying Numerical and Categorical Features:

Numerical features are extracted based on data types (int64 and float64).
Categorical features are extracted based on object data types.
Data Preprocessing:

Missing values in numerical features are filled with the mean using SimpleImputer.
Missing values in categorical features are filled with the most frequent value and encoded into one-hot format using OneHotEncoder.
A ColumnTransformer is used to apply these transformations to the respective feature types.
Splitting the Dataset:
The data is split into training and testing sets in an 80:20 ratio using train_test_split.

Model Training:

A RandomForestRegressor is selected as the predictive model.
A Pipeline is created to combine the preprocessing steps and the model.
The pipeline is trained using the training dataset.
Model Evaluation:

The model is evaluated on the test dataset using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
These metrics help determine the accuracy and performance of the model.
Feature Importance Analysis:

If supported by the model, the program extracts feature importance values.
The most influential features are displayed in a bar chart to provide insights into the model’s decision-making.
Saving the Model Pipeline:

The entire pipeline (preprocessing and trained model) is saved as a .pkl file using joblib for future use.
