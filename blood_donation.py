

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from operator import itemgetter

# Read in dataset
transfusion = pd.read_csv('/content/transfusion.data')

# Print out the first few rows of the dataset
print(transfusion.head())

# Print a concise summary of transfusion DataFrame
print(transfusion.info())

# Rename target column as 'target' for brevity
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Print out the first 2 rows
print(transfusion.head(2))

# Print target incidence proportions, rounding output to 3 decimal places
print(transfusion.target.value_counts(normalize=True).round(3))

# Split transfusion DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)

# Print out the first 2 rows of X_train
print(X_train.head(2))

# Install TPOT
!pip install tpot

# Import TPOTClassifier
from tpot import TPOTClassifier

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)

# Train the TPOT model
tpot.fit(X_train, y_train)

# AUC score for TPOT model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')

# Print variance of X_train
print(X_train.var().round(3))

# Copy X_train and X_test into new normalized datasets
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
print(X_train_normed.var().round(3))

# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the logistic regression model
logreg.fit(X_train_normed, y_train)

# AUC score for logistic regression model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')

# Sort models based on their AUC score from highest to lowest
sorted_models = sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True
)

print("\nSorted models based on AUC score:", sorted_models)