# -------------------------------------- Sklearn Random Forest Regression Example --------------------------------------

# Import stuff
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read in csv as dataframe
df = pd.read_csv('C:\\Users\\Jeremy\\Desktop\\house_prices.csv', header=0)

# set index for start of feature data (after info columns)
f_idx = 3

# get list of column names (excluding last, which is target column)
feat_names = list(df)
del feat_names[-1]
no_info_feat_names = feat_names[f_idx:]

# separate feature data from target column
X = df[feat_names]
y = df.iloc[:,-1]

# split feature data into training and test data (X is feature data, y is target column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# slice info columns for later (these should be to the left of feature data), then we remove cols from train and test
# NOTE: must choose appropriate index!
# NOTE: this is so we retain info rows for test rows
test_info_df = X_test.iloc[:,:f_idx]
X_train = X_train.iloc[:,f_idx:]
X_test = X_test.iloc[:,f_idx:]

# Apply random forest regression
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

# Calculate the root mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error: " + str(mse))
print("Root Mean Squared Error: " + str(rmse))

# rebuild dataframe to present scored data and results
test_info_df = pd.concat([test_info_df, X_test], axis=1)
test_info_df["target"] = y_test
test_info_df["pred"] = y_pred

# export results to csv
final_df = test_info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\rforest_regression_test_results.csv', index=False)
