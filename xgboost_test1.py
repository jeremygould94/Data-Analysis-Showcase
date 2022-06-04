# ############################################### XGBOOST Classifier ###################################################
# NOTE: use getdummies script to convert any category data in csv to numerical values first!

# import stuff
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# set print width to be auto
pd.options.display.width = 0

# ----------------------------------------------- setup dataframe etc --------------------------------------------------

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
X_no = df[no_info_feat_names]
y = df.iloc[:,-1]

# split feature data into training and test data (X is feature data, y is target column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# slice info columns for later (these should be to the left of feature data), then we remove cols from train and test
# NOTE: must choose appropriate index!
# NOTE: this is so we retain info rows for test rows
test_info_df = X_test.iloc[:,:f_idx]
X_train = X_train.iloc[:,f_idx:]
X_test = X_test.iloc[:,f_idx:]

# --------------------------------------------- initial XGBoost attempt ------------------------------------------------

# create XGBoost model
# use .get_params() to view parameters and tune model as needed, these are just some parameters that can be used
xgb_model = xgboost.XGBClassifier(learning_rate=0.1,
                                  max_depth=5,
                                  n_estimators=5000,
                                  substample=0.5,
                                  colsample_bytree=0.5,
                                  eval_metric="auc",
                                  verbosity=1)

eval_set = [(X_test, y_test)]

xgb_model.fit(X_train,
              y_train,
              early_stopping_rounds=10,
              eval_set=eval_set,
              verbose=True)

# evaluate initial model performance
y_train_pred = xgb_model.predict_proba(X_train)[:,1]
y_test_pred = xgb_model.predict_proba(X_test)[:,1]

print("AUC Train: {:.4f}".format(roc_auc_score(y_train, y_train_pred)))
print("AUC Test: {:.4f}".format(roc_auc_score(y_test, y_test_pred)))

# --------------------------------- set different combinations for hyperparameters -------------------------------------

# hyperparameter tuning (this is just a selection of hyperparameters than be tuned)
learning_rate_list = [0.02, 0.05, 0.1]
max_depth_list = [2, 3, 5]
n_estimators_list = [1000, 2000, 3000]

params_dict = {"learning_rate": learning_rate_list,
               "max_depth": max_depth_list,
               "n_estimators": n_estimators_list}

# print num combinations for hyperparameter tuning
num_combinations = 1
for v in params_dict.values():
    num_combinations *= len(v)

print(num_combinations)


# define function for my_roc_auc_score
def my_roc_auc_score(model, X_no, y):
    return roc_auc_score(y, model.predict_proba(X_no)[:,1])


# applying GridSearchCV
xgb_model_hp = GridSearchCV(estimator=xgboost.XGBClassifier(subsample=0.5,
                                                            colsample_bytree=0.25,
                                                            eval_metric="auc",
                                                            use_label_encoder=False),
                            param_grid=params_dict,
                            cv=2,
                            scoring=my_roc_auc_score,
                            return_train_score=True,
                            verbose=4)

xgb_model_hp.fit(X_no, y)

# inspect results of hyperparameter tuning
df_cv_results = pd.DataFrame(xgb_model_hp.cv_results_)
df_cv_results = df_cv_results[["rank_test_score",
                               "mean_test_score",
                               "mean_train_score",
                               "param_learning_rate",
                               "param_max_depth",
                               "param_n_estimators"]]
df_cv_results.sort_values(by="rank_test_score", inplace=True)
print(df_cv_results.head())

# NOTE: if needed, can plot effect of varying these parameters, or can just choose the top ranked from above
# (code to plot this not included)

# ---------------------------------------------- final run of XGBoost --------------------------------------------------

# final model - now with established best hyperparameters applied (inspect previous results and replace param values)
xgb_model_fin = xgboost.XGBClassifier(learning_rate=0.05,
                                      max_depth=5,
                                      n_estimators=3000,
                                      subsample=0.5,
                                      colsample_bytree=0.25,
                                      eval_metric="auc",
                                      verbosity=1,
                                      use_label_encoder=False)

# pass both training and testing datasets, as we want to plot AUC for both
eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model_fin.fit(X_train,
                  y_train,
                  early_stopping_rounds=20,
                  eval_set=eval_set,
                  verbose=True)

# evaluate final model performance
y_train_pred = xgb_model_fin.predict_proba(X_train)[:,1]
y_test_pred = xgb_model_fin.predict_proba(X_test)[:,1]

print("AUC Train: {:.4f}".format(roc_auc_score(y_train, y_train_pred)))
print("AUC Test: {:.4f}".format(roc_auc_score(y_test, y_test_pred)))

# NOTE: again, there is more code that can be used to plot the varying performance based on number of trees etc
# (code not included here)

# evaluate feature importance
df_feat_imp = pd.DataFrame({"Feature": no_info_feat_names,
                            "Importance": xgb_model_fin.feature_importances_}).sort_values(by="Importance",
                                                                                           ascending=False)

# print selection of most import features
print(df_feat_imp[:20])

# ------------------------------------------------- output results -----------------------------------------------------

# rebuild dataframe to present scored data and results
for column in X_test:
    test_info_df[column] = X_test[column]

test_info_df["target"] = y_test
test_info_df["pred"] = y_test_pred
test_info_df["pred_binary"] = round(test_info_df["pred"])

# export results to csv
final_df = test_info_df
final_df.to_csv('C:\\Users\\Jeremy\\Desktop\\xgboost_test_results.csv', index=False)
