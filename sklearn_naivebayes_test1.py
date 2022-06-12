# --------------------------------------- Sklearn Naive Bayes Classifier Example ---------------------------------------
"""
Info:
> Naive Bayes is a machine learning method you can use to predict the likelihood that an event will occur given
  evidence that's present in your data.
> In stastics, this often referred to as 'conditional probability' and expressed as...
  >> 'The probability of B given A is the probability of A AND B over the probability of A'
> There are three types of Naive Bayes models
  >> Bernoulli - good for making predictions from binary features.
  >> Gaussian - good for making predictions from normally distributed features.
  >> Multinomial - good for when your features are categorical or continuous and describe descrete frequency counts
     (e.g. word counts).
> Common use cases
  >> Spam detection
  >> Customer classification
  >> Credit risk prediction
  >> Health risk prediction
> Assumptions when using Naive Bayes
  >> Predictors are independent of each other
  >> 'A priori' assumption - meaning we assume that past conditions still hold true. If present circumstances have
     changed, we will get incorrect results when applying historical values.
"""

# Import stuff
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --------------------------------------------------- Setup data -------------------------------------------------------

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

# NOTE: How our data is and what we are trying to do should indicate which type of Naive Bayes we should use.
#       We will outline how to do all three.

# ---------------------------------------------------- Bernoulli -------------------------------------------------------

# Apply Bernoulli
BernNB = BernoulliNB(binarize=True)  # (Binarize argument can accept decimal. Experiment with different values)
BernNB.fit(X_train, y_train)
y_BernNB_pred = BernNB.predict(X_test)

# rebuild dataframe to present scored data and results
bernNB_df = pd.concat([test_info_df, X_test], axis=1)
bernNB_df["target"] = y_test
bernNB_df["bern_pred"] = y_BernNB_pred

# export results to csv
bernNB_df.to_csv('C:\\Users\\Jeremy\\Desktop\\naivebayes_bern_test_results.csv', index=False)

# Display the Confusion Matrix (TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative)
# NOTE: ALWAYS CHECK MATRIX AGAINST DATA!
con_mat = confusion_matrix(y_test, y_BernNB_pred)
TN = con_mat[0][0]
FP = con_mat[0][1]
FN = con_mat[1][0]
TP = con_mat[1][1]
con_mat_df = pd.DataFrame(con_mat)
new_con_mat_df = pd.DataFrame(["TN", "FN"], columns=["c1"])
new_con_mat_df["c2"] = con_mat_df.iloc[:,0]
new_con_mat_df["c3"] = ["FP", "TP"]
new_con_mat_df["c4"] = con_mat_df.iloc[:,1]
print("Bernoulli Confusion Matrix:")
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
print("________________________________________")
print("")

# ---------------------------------------------------- Gaussian --------------------------------------------------------

# Apply Gaussian
GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
y_GaussNB_pred = GausNB.predict(X_test)

# rebuild dataframe to present scored data and results
gausNB_df = pd.concat([test_info_df, X_test], axis=1)
gausNB_df["target"] = y_test
gausNB_df["gaus_pred"] = y_GaussNB_pred

# export results to csv
gausNB_df.to_csv('C:\\Users\\Jeremy\\Desktop\\naivebayes_gaus_test_results.csv', index=False)

# Display the Confusion Matrix (TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative)
# NOTE: ALWAYS CHECK MATRIX AGAINST DATA!
con_mat = confusion_matrix(y_test, y_GaussNB_pred)
TN = con_mat[0][0]
FP = con_mat[0][1]
FN = con_mat[1][0]
TP = con_mat[1][1]
con_mat_df = pd.DataFrame(con_mat)
new_con_mat_df = pd.DataFrame(["TN", "FN"], columns=["c1"])
new_con_mat_df["c2"] = con_mat_df.iloc[:,0]
new_con_mat_df["c3"] = ["FP", "TP"]
new_con_mat_df["c4"] = con_mat_df.iloc[:,1]
print("Gaussian Confusion Matrix:")
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
print("________________________________________")
print("")


# --------------------------------------------------- Multinomial ------------------------------------------------------

# Apply Multinomial
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
y_MultiNB_pred = MultiNB.predict(X_test)

# rebuild dataframe to present scored data and results
multiNB_df = pd.concat([test_info_df, X_test], axis=1)
multiNB_df["target"] = y_test
multiNB_df["multi_pred"] = y_MultiNB_pred

# export results to csv
multiNB_df.to_csv('C:\\Users\\Jeremy\\Desktop\\naivebayes_multi_test_results.csv', index=False)

# Display the Confusion Matrix (TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative)
# NOTE: ALWAYS CHECK MATRIX AGAINST DATA!
con_mat = confusion_matrix(y_test, y_MultiNB_pred)
TN = con_mat[0][0]
FP = con_mat[0][1]
FN = con_mat[1][0]
TP = con_mat[1][1]
con_mat_df = pd.DataFrame(con_mat)
new_con_mat_df = pd.DataFrame(["TN", "FN"], columns=["c1"])
new_con_mat_df["c2"] = con_mat_df.iloc[:,0]
new_con_mat_df["c3"] = ["FP", "TP"]
new_con_mat_df["c4"] = con_mat_df.iloc[:,1]
print("Multinomial Confusion Matrix:")
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
