# --------------------------------------- Sklearn K-Means for Clustierng Example ---------------------------------------
"""
Info:
> K-Means Clustering is an Unsupervised Learning algorithm, used to group an unlabeled dataset into different
  clusters/subsets (where K is the number of clusters).
> This can be particularly useful when analysing customer behaviour, as there can be different trends atrributed
  to different demographics.
"""
# NOTE: Correlation/scatter matrix doesn't work well with onehotencoded data

# Import stuff

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import seaborn as sns

# set print width to be auto
pd.options.display.width = 0

# Read in csv as dataframe
df = pd.read_csv('C:\\Users\\Jeremy\\Desktop\\catdog_numerical_data.csv', header=0)

# set index for start of feature data (after info columns)
f_idx = 3

# get list of column names (excluding last, which is target column)
feat_names = list(df)
del feat_names[-1]
no_info_feat_names = feat_names[f_idx:]

# separate feature data from target column
X = df[feat_names]
y = df.iloc[:,-1]

# slice info columns for later (these should be to the left of feature data)
# NOTE: must choose appropriate index!
info_df = X.iloc[:,:f_idx]
feat_data_df = X.iloc[:,f_idx:]

# Check descriptive stats
print(feat_data_df.describe())

# Plot feature correlation
g = pd.plotting.scatter_matrix(df[["Weight(kg)", "Length(cm)", "Height(cm)"]],
                               figsize=(10,10),
                               marker='o',
                               hist_kwds={'bins': 10},
                               s=60,
                               alpha=0.8)
plt.show()

# Apply k-means with 5 clusters (here we have selected 2 columns)
kmeans = cluster.KMeans(n_clusters=5, init="k-means++")
keans = kmeans.fit(df[["Weight(kg)", "Length(cm)"]])

# Display cluster centers
print(kmeans.cluster_centers_)

# Attach target and clusters to the original data (here we have 5 clusters, so labels will be 0-4)
feat_data_df["Target"] = y
feat_data_df["Clusters"] = kmeans.labels_

# Concat feature data to right of info data
cluster_df = pd.concat([info_df, feat_data_df], axis=1)

# Inspect cluster df
print(cluster_df.head())
print(cluster_df["Clusters"].value_counts())

# Plot cluster on chart
sns.scatterplot(x="Weight(kg)", y="Length(cm)", hue="Clusters", data=cluster_df)
plt.show()
