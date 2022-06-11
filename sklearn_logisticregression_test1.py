# ---------------------------------------- Sklearn Logistic Regression Example -----------------------------------------
# NOTE: Despite the name "Logistic Regression", this algortihm is actually used for binary classification.

# Import Stuff
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Generate the dataset
X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)

# Visualize the data
plt.scatter(X, y, c=y, cmap="rainbow")
plt.title("Scatter Plot of Logistic Regression")
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Perform Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make prediction using the model
y_pred = log_reg.predict(X_test)

# Display the Confusion Matrix (TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative)
# NOTE: ALWAYS CHECK MATRIX AGAINST DATA!
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
