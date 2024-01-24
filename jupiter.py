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
        """
        Define constants, including gravitational constant and moon mass.
        """
        self.G = 6.67e-11
        # Calculate moon mass using Kepler's Third Law and add an "estimated_mass" column to the data frame
        self.data["estimated_mass"] = (4 * np.pi**2 * self.data["distance_km"]**3) / (self.data["period_days"]**2 * self.G)
   
    def summary_statistics(self):
        """
        Calculate and display summary statistics for numerical columns.
        """
        numerical_cols = self.data.select_dtypes(include=['number'])
        sns.set_theme(style="whitegrid", palette="pastel")  # Set a visually appealing theme

        for col in numerical_cols:
            sns.histplot(data=self.data, x=col)
            plt.title(f"Distribution of {col}")
            plt.show()
            
    def correlations(self):
        """
        Calculate and display correlations between numerical columns.
        """
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
        """
        Provide descriptive information about a specified column.

        Parameters:
        - column_name (str): Name of the column to describe.
        """
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
        """
        Display unique items in the 'moon' column.
        """
        moons = self.data["moon"].unique()
        print("Moon Items:")
        for moon in moons:
            print(moon)

