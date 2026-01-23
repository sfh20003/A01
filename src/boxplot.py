from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

#First had to download matplotlib and Sklearn in terminal
#Box plot of median household value 
import matplotlib.pyplot as plt
plt.figure
plt.boxplot(df["MedHouseVal"])
plt.title("Boxplot of Median House Value")
plt.ylabel("Median House (100,000s)")
plt.show()
