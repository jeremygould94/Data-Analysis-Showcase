# ---------- Decision tree using sklearn ----------
# NOTE: use getdummies script to convert any category data in csv to numerical values first!

# import stuff
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt

# Read in csv as dataframe
df = pd.read_csv('C:\\Users\\Jeremy\\Desktop\\post_getdummies_csv1.csv', header=0)

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

# apply decision tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# define ML accuracy function
def ml_accuracy(y_true, y_prediction):
    accuracy = np.sum(y_true == y_prediction) / len(y_true)
    return accuracy


# print accuracy
acc = ml_accuracy(y_test, y_pred)
print("Accuracy:", acc)

# rebuild dataframe to present scored data and results
test_info_df = pd.concat([test_info_df, X_test], axis=1)
test_info_df["target"] = y_test
test_info_df["pred"] = y_pred

# export results to csv
final_df = test_info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\dtree_test_results.csv', index=False)

# plot decision tree
target_names = ["0: Dog", "1: Cat"]  # uncomment and name as appropriate
fig = plt.figure(figsize=(100, 100))
_ = tree.plot_tree(clf,
                   feature_names=no_info_feat_names,
                   class_names=target_names,
                   filled=True)

fig.savefig('C:\\Users\\Jeremy\\Desktop\\dtree_test_fig.png')
