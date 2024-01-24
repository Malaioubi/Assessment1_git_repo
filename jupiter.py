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
    """
    Class for analyzing moon data and estimating Jupiter's mass.

    Attributes:
        conn (sqlite3.Connection): Connection to the SQLite database.
        data (pd.DataFrame): DataFrame containing moon data.
        G (float): Gravitational constant.

    Methods:
        __init__(database_path: str)
        __del__()
        load_data()
        define_constants()
        summary_statistics()
        correlations()
        describe_column(column_name: str)
        display_moons()
        scatter_plot(x_column: str, y_column: str)
        extract_moon_data(moon: str) -> pd.Series
        prepare_data() -> pd.DataFrame
        train_model(modeling_data: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]
        evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series)
        estimate_jupiter_mass(model: LinearRegression)
    """

    def __init__(self, database_path: str):
        """
        Constructor to initialize the object.

        Parameters:
        - database_path (str): Path to the SQLite database containing moon data.
        """
        self.conn = sqlite3.connect(database_path)
        self.load_data()  # Load moon data from the database
        self.define_constants()  # Define constants including gravitational constant and moon mass

    def __del__(self):
        """
        Destructor to ensure the database connection is closed when the object is deleted.
        """
        self.conn.close()

    def load_data():
        """
        Load moon data from the database.
        """
        self.data = pd.read_sql_query("SELECT * FROM moons", self.conn)

    def define_constants():
        """
        Define constants, including gravitational constant and moon mass.
        """
        self.G = 6.67e-11
        # Calculate moon mass using Kepler's Third Law and add an "estimated_mass" column to the data frame
        self.data["estimated_mass"] = (4 * np.pi**2 * self.data["distance_km"]**3) / (self.data["period_days"]**2 * self.G)

    def summary_statistics():
        """
        Calculate and display summary statistics for numerical columns.
        """
        numerical_cols = self.data.select_dtypes(include=['number'])
        sns.set_theme(style="darkgrid")  # Set a visually appealing theme

        for col in numerical_cols:
            sns.histplot(data=self.data, x=col)
            plt.title(f"Distribution of {col}")
            plt.show()

    def correlations():
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

    def describe_column(column_name: str):
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
            
    def display_moons():
        """
        Display unique items in the 'moon' column.
        """
        moons = self.data["moon"].unique()
        print("Moon Items:")
        for moon in moons:
            print(moon)

    def scatter_plot(x_column: str, y_column: str):
        """
        Generate a scatter plot of two specified columns.

        Parameters:
        - x_column (str): Name of the column for the x-axis.
        - y_column (str): Name of the column for the y-axis.
        """
        plt.figure(figsize=(8, 6))  # Set appropriate figure size
        plt.scatter(self.data[x_column], self.data[y_column])

        # Label axes and add title
        plt.xlabel(x_column.capitalize())
        plt.ylabel(y_column.capitalize())
        plt.title(f"Scatter Plot of {y_column} vs. {x_column}")

        plt.grid(True)  # Add grid for better readability
        plt.show()

    def extract_moon_data(moon: str) -> pd.Series:
        """
        Extract data for a specific moon.

        Parameters:
        - moon (str): Name of the moon.

        Returns:
        - pd.Series: Data for the specified moon.
        """
        moon_data = self.data[self.data["moon"] == moon]
        return moon_data.iloc[0]  # Return the first row if multiple matches and allows for a better visualization of data

    def prepare_data() -> pd.DataFrame:
        """
        Prepare data for modeling.

        Returns:
        - pd.DataFrame: Prepared data for modeling.
        """
        # Calculate T^2 and a^3, handle missing values, and convert units
        self.data["T2"] = (self.data["period_days"] * 24 * 60 * 60) ** 2  # Convert days to seconds
        self.data["a3"] = (self.data["distance_km"] * 1000) ** 3  # Convert km to meters
        self.data = self.data.dropna(subset=["T2", "a3"])  # Drop rows with missing values

        # Create DataFrame for modeling
        modeling_data = self.data[["moon", "T2", "a3"]]
        return modeling_data

    def train_model(modeling_data: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
        """
        Train a linear regression model using train_test_split.

        Parameters:
        - modeling_data (pd.DataFrame): Data for modeling.

        Returns:
        - tuple: Trained model, X_test, and y_test.
        """
        X = modeling_data[["T2"]]
        y = modeling_data["a3"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # Set random_state for reproducibility

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, X_test, y_test  # Return model, X_test, and y_test

    def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the trained model using test data.

        Parameters:
        - model (LinearRegression): Trained linear regression model.
        - X_test (pd.DataFrame): Test data for the features.
        - y_test (pd.Series): Test data for the target variable.
        """
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        r_squared = r2_score(y_test, predictions)

        # Print results and visualize residuals (optional)
        print("Mean Squared Error (testing set):", mse)
        print("R-squared (testing set):", r_squared)

    def estimate_jupiter_mass(model: LinearRegression):
        """
        Calculate and display estimated Jupiter's mass, differences from literature, and Jupiter-to-Earth mass ratio.

        Parameters:
        - model (LinearRegression): Trained linear regression model.
        """
        jupiter_mass_kg = (4 * (np.pi ** 2)) * model.coef_[0] / self.G

        # Comparison with literature value and with Earth
        literature_jupiter_mass_kg = 1.8982e27  # From [Source reference]
        literature_earth_mass_kg = 5.972e24  # Define Earth's mass in kg

        absolute_difference = abs(jupiter_mass_kg - literature_jupiter_mass_kg)
        percentage_difference = (absolute_difference / literature_jupiter_mass_kg) * 100

        print("Estimated Jupiter mass:", jupiter_mass_kg, "kg")
        print("Difference from literature:", absolute_difference, "kg")
        print("Percentage difference:", percentage_difference, "%")

        estimated_jupiter_earth_ratio = jupiter_mass_kg / literature_earth_mass_kg
        print(f"Estimated Jupiter-to-Earth mass ratio: {estimated_jupiter_earth_ratio:.2f}")   
