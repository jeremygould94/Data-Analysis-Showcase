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

# set top category limit - only preserve detail for cateogry data in top X per column (for effective onehotencoding)
# NOTE: to preserve all category data, set to 0
top_cat = 10

# loop through columns, retaining only top X entries in each category column, replacing others with "OOS" (out of scope)
if top_cat != 0:
    for column in new_df:
        if new_df[column].dtypes == "object":
            top_cat_list = new_df[column].value_counts().index.tolist()[:top_cat]
            new_df.loc[~new_df[column].isin(top_cat_list), column] = "OOS"

# loop through new dataframe columns, create dummies for columns containing categorical data
for column in new_df:
    if new_df[column].dtypes == "object":
        new_df = pd.get_dummies(new_df, columns=[column])

# add last (target) column back onto right of df
new_df["target"] = df_target

# add new dataframe to right of info dataframe
info_df = pd.concat([info_df, new_df], axis=1)

# rename for clarity and export dataframe to csv
final_df = info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\post_getdummies_csv1.csv', index=False)
