# ==============================
# Weather Data Analytics Project
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("weatherHistory.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

# --------------------------
# Data Cleaning
# --------------------------

print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates()

# Convert Date

df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])

# --------------------------
# Basic Statistics
# --------------------------

print(df.describe())

# --------------------------
# Temperature Trend
# --------------------------

plt.figure(figsize=(12,5))

plt.plot(df["Temperature (C)"][:500])

plt.title("Temperature Trend")

plt.xlabel("Record")

plt.ylabel("Temperature")

plt.grid(True)

plt.show()

# --------------------------
# Humidity Distribution
# --------------------------

plt.figure(figsize=(8,5))

plt.hist(df["Humidity"], bins=20)

plt.title("Humidity Distribution")

plt.xlabel("Humidity")

plt.ylabel("Frequency")

plt.show()

# --------------------------
# Correlation
# --------------------------

numeric = df.select_dtypes(include=np.number)

print("\nCorrelation Matrix")

print(numeric.corr())

# --------------------------
# Machine Learning
# Predict Temperature
# --------------------------

features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Pressure (millibars)",
    "Visibility (km)"
]

X = df[features]

y = df["Temperature (C)"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nModel Accuracy")

print("R2 Score :", r2_score(y_test, prediction))

print("MAE :", mean_absolute_error(y_test, prediction))

# --------------------------
# Predict Sample
# --------------------------

sample = [[
    0.70,
    15,
    200,
    1012,
    10
]]

result = model.predict(sample)

print("\nPredicted Temperature")

print(result[0])
