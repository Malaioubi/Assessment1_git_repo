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
