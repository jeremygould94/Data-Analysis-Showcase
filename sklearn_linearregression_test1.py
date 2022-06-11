# --------------------------------------- Sklearn Linear Regression CSV Example ----------------------------------------

# Import stuff
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read in csv as dataframe
df = pd.read_csv('C:\\Users\\Jeremy\\Desktop\\linear_test_data.csv', header=0)

# Apply processing to dataframe (could filter out NaN rows or outliers etc)
filtered_df = df[~np.isnan(df["y"])]  # this removes rows with NaN in them

# Reshaping
X_y = np.array(filtered_df)
X, y = X_y[:,0], X_y[:,1]
X, y = X.reshape(-1,1), y.reshape(-1, 1)

# Apply Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Get gradient of fitted line
m = lin_reg.coef_
m_rnd = round(m[0][0],2)

# Get y-Intercept of the Line
b = lin_reg.intercept_
b_rnd = round(b[0],2)

# Get Predictions for original X values (you can also get predictions for new data)
predictions = lin_reg.predict(X)

# Display equation in form "y = mx + c"
print("Equation: y = " + str(m_rnd) + "x + " + str(b_rnd))

# Plot the Original Model (Black) and Predictions (Blue)
plt.scatter(X, y,  color='black')
plt.plot(X, predictions, color='blue',linewidth=3)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.show()
