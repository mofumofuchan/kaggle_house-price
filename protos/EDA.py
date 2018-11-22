# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.1
# ---

# +
import numpy as np
import scipy as sp
import sklearn as skl
import pandas as pd
import pandas_profiling

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# -

#!ls input
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# +
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

for i in tqdm(range(df_train.shape[1])):
    if df_train.iloc[:, i].dtypes == object:
        labeler = LabelEncoder()
        labeler.fit(list(df_train.iloc[:,i].values) + list(df_test.iloc[:,i].values))
        df_train.iloc[:,i] = labeler.transform(list(df_train.iloc[:,i].values))
        df_test.iloc[:,i] = labeler.transform(list(df_test.iloc[:,i].values))
# -

import missingno as msno
# %matplotlib inline
msno.matrix(df_train, labels=True, figsize=(40,12))



pandas_profiling.ProfileReport(df_train)

train_ID = df_train["Id"]
test_ID = df_test["Id"]

# +
y_train = df_train["SalePrice"]
X_train = df_train.drop(["Id", "SalePrice"], axis=1)

X_test = df_test.drop(["Id"], axis=1)
# -

ax = sns.distplot(y_train)
plt.show()

y_train_log = np.log(y_train)

sns.distplot(y_train_log)

X_train = X_train.drop(["LotFrontage", "MasVnrArea", "GarageYrBlt"], axis=1)
X_train = X_train.fillna(X_train.median())

# +
fig = plt.figure(figsize=(12,12))
for i in np.arange(42):
    ax = fig.add_subplot(7,6,i+1)
    sns.regplot(x=X_train.iloc[:,i], y=y_train)

plt.tight_layout()
plt.show()

# +
import xgboost as xgb

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(x_train, y_train_log)

X_test = X_test.fillna(X_test.median())

y_test = np.exp(xgb_reg.predict(X_test))
# -

submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": y_test
})
submission.to_csv("houseprice.csv", index=False)
