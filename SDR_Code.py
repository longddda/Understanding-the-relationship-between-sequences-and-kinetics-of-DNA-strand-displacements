#This code is provided by longda
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Data preprocessing
df = pd.read_csv("DATASET.csv")
#del df['name']
#del df['sequence(invader)']
#del df["Rate Constant L/(s*nmol)"]

# Modify the column name
df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'Label']
# Replacement of column names
df.reset_index(drop=True, inplace=True)
# Get variables from the first to the sixteenth.
feat_cols = ['X' + str(i + 1) for i in range(16)]
train_feats = df[feat_cols]
Y_train = df['Label']

# Data standardisation
feat_scalar = StandardScaler().fit(train_feats)
label_scaler = StandardScaler().fit(Y_train.values.reshape(-1, 1))

scale_data = df.copy()
scale_data[feat_cols] = feat_scalar.transform(df[feat_cols])
scale_data['Label'] = Y_train.values.reshape(-1, 1)

# Viewing Feature Importance Using Random Forests
feat_labels = df.columns[:-1]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(scale_data[feat_cols], scale_data['Label'])
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(scale_data[feat_cols].shape[1]):
    print("%2d)%-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.figure(figsize=(10, 6), dpi=300)
plt.title('Feature Importance', fontdict={'family': 'Arial', 'size': 32})
p1 = plt.bar(range(scale_data[feat_cols].shape[1]), importances[indices].round(3), align='center')
plt.bar_label(p1, label_type='edge', fontproperties='Arial', fontsize=13)
plt.xticks(range(scale_data[feat_cols].shape[1]), feat_labels[indices], rotation=90, fontproperties='Arial',
           fontsize=25)
plt.yticks(fontproperties='Arial', fontsize=25)
plt.xlim([-1, scale_data[feat_cols].shape[1]])
plt.tight_layout()
plt.show()

# 1. Using logistic regression algorithms
# Recursive feature elimination as well as cross-validation are used to filter features: logistic is used as an estimator
rfecv = RFECV(estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                           C=10, class_weight='balanced',
                                           random_state=42, tol=0.0001), cv=3, n_jobs=-1,
              scoring='accuracy')
train_feats = scale_data[feat_cols]
train_label = scale_data['Label']
rfecv.fit(train_feats, train_label)
# Output the more important features
keep_index = np.where(rfecv.support_ == True)[0]
selected_features = train_feats.columns[keep_index]
#print(f"Selected Features: {selected_features}")

# Slicing the dataset
X_train, X_test, y_train, y_test = train_test_split(scale_data[selected_features], scale_data['Label'],
                                                    test_size=0.30, random_state=10, stratify=scale_data['Label'])

# Screening of hyperparameters using cross-validation
parameters = {'C': [0.1, 0.3, 0.5, 1, 3, 5],
              'max_iter': [50, 100, 300, 500, 1000],
              'tol': [1e-3, 5e-3, 1e-1, 3e-1, 5e-1]}
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=3, class_weight='balanced',
                           random_state=42, tol=0.1)
classifier_1 = GridSearchCV(model, parameters, cv=3)
classifier_1.fit(X_train, y_train)
y_pred = classifier_1.predict(X_test)
y_train_pred = classifier_1.predict(X_train)
print(f"Logistic Regression Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Selected Features: {selected_features}")

# 2. Using support vector machine algorithms
# Recursive feature elimination as well as cross-validation are used to filter features: using SVC as an estimator
rfecv = RFECV(estimator=SVC(kernel='linear'), cv=3, n_jobs=-1, scoring='accuracy')
train_feats = scale_data[feat_cols]
train_label = scale_data['Label']
rfecv.fit(train_feats, train_label)
# Output the more important features
keep_index = np.where(rfecv.support_ == True)[0]
selected_features = train_feats.columns[keep_index]
#print(f"Selected Features: {selected_features}")

# Slicing the dataset
X_train, X_test, y_train, y_test = train_test_split(scale_data[selected_features], scale_data['Label'],
                                                    test_size=0.3, random_state=40, stratify=scale_data['Label'])

# Screening of hyperparameters using cross-validation
parameters = [
    {'C': [1, 5, 11, 15, 19], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100], 'kernel': ['rbf']},
    {'C': [1, 5, 11, 15, 19], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [1, 5, 11, 15, 19], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100], 'kernel': ['sigmoid']},
    {'C': [1, 5, 11, 15, 19], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100], 'kernel': ['sigmoid']},
    {'C': [1, 5, 11, 15, 19], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100], 'kernel': ['poly']}
]

model = SVC()
classifier_2 = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
classifier_2.fit(X_train, y_train)
y_pred = classifier_2.predict(X_test)
y_train_pred = classifier_2.predict(X_train)
print(f"SVM Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Selected Features: {selected_features}")

# 3.Using decision tree algorithm
# Slicing the dataset
X_train, X_test, y_train, y_test = train_test_split(scale_data[['X2', 'X5', 'X14']], scale_data['Label'],
                                                    test_size=0.25, random_state=30, stratify=scale_data['Label'])

# Screening of hyperparameters using cross-validation
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [30, 50, 60, 100],
    'min_samples_leaf': [2, 3, 5, 10],
    'min_impurity_decrease': [0.001, 0.002, 0.1, 0.2, 0.5],
    'splitter': ['best', 'random'],
    'min_samples_leaf': range(1, 50, 5)
}
model = tree.DecisionTreeClassifier(random_state=1)
classifier_3 = GridSearchCV(model, parameters, cv=3)
classifier_3.fit(X_train, y_train)
y_pred = classifier_3.predict(X_test)
y_train_pred = classifier_3.predict(X_train)
print(f"Decision Tree Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Selected Features: {['X2', 'X5', 'X14']}")
