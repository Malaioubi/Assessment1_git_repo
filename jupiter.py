# Import necessary libraries
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class Moons:
    
    def __init__(self, database_path):
        self.conn = sqlite3.connect(database_path)
        self.load_data()  # Load moon data from the database
        self.define_constants()  # Define constants including gravitational constant and moon mass
      
    def __del__(self):
        self.conn.close()    
        
    def define_constants(self):
        self.G = 6.67e-11
        # Calculate moon mass using Kepler's Third Law and add an "estimated_mass" column to the data frame
        self.data["estimated_mass"] = (4 * np.pi**2 * self.data["distance_km"]**3) / (self.data["period_days"]**2 * self.G)
        
# Methods for exploratory analysis:
    def summary_statistics(self):
        numerical_cols = self.data.select_dtypes(include=['number'])
        sns.set_theme(style="whitegrid", palette="pastel")  # Set a visually appealing theme

        for col in numerical_cols:
            sns.histplot(data=self.data, x=col)
            plt.title(f"Distribution of {col}")
            plt.show()
            
    def correlations(self):
        correlations = self.data.corr(numeric_only=True)
        sns.set_theme(style="darkgrid")  # Set a visually appealing theme

        # Create a visually rich correlation heatmap
        sns.heatmap(
            correlations,
            annot=True,  # Display correlation values within cells
            cmap="coolwarm",  # Colormap for heatmap
            fmt=".2f",  # Format correlation values to two decimal places
            linewidths=0.5,  # Adjust cell borders for clarity
        )
        plt.title("Correlation Heatmap")
        plt.show()
        
    def describe_column(self, column_name):
        if column_name in self.data.columns:
            column = self.data[column_name]

            # Basic information
            print("Column Name:", column_name)
            print("Data Type:", column.dtype)
            print("Number of Values:", column.count())
            print("Number of Unique Values:", column.nunique())

            # Descriptive statistics (if numerical)
            if pd.api.types.is_numeric_dtype(column):
                print(column.describe())
            else:
                # Frequency counts for non-numerical columns
                print("Most Frequent Values:")
                print(column.value_counts().head())
        else:
            print("Column not found in the data.")
            
    def display_moons(self):
        moons = self.data["moon"].unique()
        print("Moon Items:")
        for moon in moons:
            print(moon)
            
    def extract_moon_data(self, moon):
        moon_data = self.data[self.data["moon"] == moon]
        return moon_data.iloc[0]  # Return the first row if multiple matches and allows for a better visualization of data

    def scatter_plot(self, x_column, y_column):
        sns.set_theme(style="whitegrid", palette="pastel")
        plt.figure(figsize=(8, 6))  # Set appropriate figure size
        plt.scatter(self.data[x_column], self.data[y_column])
        sns.regplot(x=self.data[x_column], y=self.data[y_column], scatter=False, color='red')

        # Label axes and add title
        plt.xlabel(x_column.capitalize())
        plt.ylabel(y_column.capitalize())
        plt.title(f"Scatter Plot of {y_column} vs. {x_column}")

        plt.grid(True)  # Add grid for better readability
        plt.show()

# Estimation of Jupiter's mass:

    # Prepare data for modeling
    def prepare_data(self):
        # Calculate T^2 and a^3, handle missing values, and convert units
        self.data["T2"] = (self.data["period_days"] * 24 * 60 * 60) ** 2  # Convert days to seconds
        self.data["a3"] = (self.data["distance_km"] * 1000) ** 3  # Convert km to meters
        self.data = self.data.dropna(subset=["T2", "a3"])  # Drop rows with missing values

        # Create DataFrame for modeling
        modeling_data = self.data[["moon", "T2", "a3"]]
        return modeling_data
        # Train a linear regression model using train_test_split
    def train_model(self, modeling_data):
        X = modeling_data[["T2"]]
        y = modeling_data["a3"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # Set random_state for reproducibility

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, X_test, y_test  # Return model, X_test, and y_test

    # Evaluate the trained model using test data
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        r_squared = r2_score(y_test, predictions)

        # Print results and visualize residuals (optional)
        print("Mean Squared Error (testing set):", mse)
        print("R-squared (testing set):", r_squared)




