# ---------- Apply getdummies to category data (to transform into numerical) ----------
# Import stuff
import pandas as pd

# Read in csv as dataframe
df = pd.read_csv('C:\\Users\\Jeremy\\Desktop\\pre_getdummies_csv1.csv', header=0)

# slice off last column (target) and save for later
df_target = df.iloc[:,-1]
new_df = df.iloc[:,:-1]

# set index for start of feature data (after info columns)
f_idx = 3

# slice off info columns to save for later (these should be to the left of feature data)
# NOTE: must choose appropriate index!
info_df = new_df.iloc[:,:f_idx]
new_df = new_df.iloc[:,f_idx:]

# loop through new dataframe columns, create dummies for columns containing categorical data
for column in new_df:
    if new_df[column].dtypes == "object":
        new_df = pd.get_dummies(new_df, columns=[column])

# add last (target) column back onto right of df
new_df["target"] = df_target

# add new dataframe to right of info dataframe
for column in new_df:
    info_df[column] = new_df[column]

# rename for clarity and export dataframe to csv
final_df = info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\post_getdummies_csv1.csv', index=False)
