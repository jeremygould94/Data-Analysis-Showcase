# ------------------------------ Sklearn KNN (K Nearest Neighbour) Classification Example ------------------------------
"""
Info:
> K-NN is a supervised classifier that memorizes observations from within labeled test set to predict classification
  labels for new, unlabled observations.
> K-NN makes predictions based on how similar training observations are to the new, incoming observations.
> The more similar the obserbation's values, the more likely they will be classified with the same label.

K-NN Model Assumptions:
> Dataset has little noise.
> Dataset is labelled.
> Dataset only contains relevant features.
> Avoid using k-NN on large datasets, as it will probably take a long time.
"""

# Import stuff
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

# Apply KNN
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# rebuild dataframe to present scored data and results
test_info_df = pd.concat([test_info_df, X_test], axis=1)
test_info_df["target"] = y_test
test_info_df["pred"] = y_pred

# export results to csv
final_df = test_info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\knn_test_results.csv', index=False)

# Display the Confusion Matrix (TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative)
# NOTE: ALWAYS CHECK MATRIX AGAINST DATA!
con_mat = confusion_matrix(y_test, y_pred)
TN = con_mat[0][0]
FP = con_mat[0][1]
FN = con_mat[1][0]
TP = con_mat[1][1]
con_mat_df = pd.DataFrame(con_mat)
new_con_mat_df = pd.DataFrame(["TN", "FN"], columns=["c1"])
new_con_mat_df["c2"] = con_mat_df.iloc[:,0]
new_con_mat_df["c3"] = ["FP", "TP"]
new_con_mat_df["c4"] = con_mat_df.iloc[:,1]
print("Confusion Matrix:")
print(new_con_mat_df.to_string(index=False, header=False))
print("")

# print ML accuracy
acc = (TN + TP) / (TN + FP + TP + FN)
print(f"Test Accuracy: {acc:.3f}")

# print precision (Precision = 1 means no false positives)
prec = TP / (TP + FP)
print(f"Test Precision: {prec:.3f}")

# print recall (Recall = 1 means no false negatives)
recall = TP / (TP + FN)
print(f"Test Recall: {recall:.3f}")

# print F1 score (F1 score becomes 1 when precision and recall are both 1)
f1_score = 2 * (prec * recall) / (prec + recall)
print(f"Test F1 Score: {f1_score:.3f}")
