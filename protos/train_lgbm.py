#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm

from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import pickle
import os

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = "result_tmp/"
SAMPLE_SUBMIT_FILE = "../input/sample_submission.csv"
LOG_FILENAME = "train_lgbm.py.log"
PARAMS_FILENAME = "param.pickle"

def rmse_func(pred, y):
    return np.sqrt(mean_squared_error(y, pred))


if __name__ == "__main__":

    log_fmt = Formatter("%(asctime)s %(name)s  %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ")
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + LOG_FILENAME, "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info("start")

    df = load_train_data()

    X_train = df.drop(["SalePrice", "Id"], axis=1)
    y_train = df["SalePrice"].values

    y_train = np.log(y_train)

    use_cols = X_train.columns.values

    logger.info("train columns: {} {}".format(use_cols.shape, use_cols))
    logger.info("data preparation end {}".format(X_train.shape))


    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {"max_depth": [3, 5, 7],
                  "learning_rate": [0.1],
                  "min_child_weight": [3, 5, 10],
                  "n_estimators": [10000],
                  "colsample_bytree": [0.8, 0.9],
                  "reg_alpha": [0, 0.1],
                  "n_jobs": [4],
                  "random_state": [0]}

    min_error = 100
    min_params = None
    for params in tqdm(list(ParameterGrid(param_grid=all_params))):
        logger.info("params: {}".format(params))
        list_rmse = []
        list_mae = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            trn_x = X_train.iloc[train_idx, :].values
            val_x = X_train.iloc[valid_idx, :].values

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = lgbm.LGBMRegressor(**params)
            clf.fit(trn_x,
                    trn_y,
                    eval_set=[(val_x, val_y)],
                    early_stopping_rounds=100,
                    eval_metric=["rmse"],
                    verbose=False)

            pred = clf.predict(val_x, num_iteration=clf.best_iteration_)
            rmse_val = rmse_func(val_y, pred)

            list_rmse.append(rmse_val)
            logger.debug("    rmse: {}".format(rmse_val))

            break  # FIXME trial

        mean_rmse = np.mean(list_rmse)

        logger.info("rmse: {}".format(mean_rmse))
        if min_error > mean_rmse:
            min_error = mean_rmse
            min_params = params
        logger.info("current minerror: {}, params: {}".format(min_error,
                                                              min_params))

    logger.info("Save best params")
    with open(os.path.join(DIR, PARAMS_FILENAME), mode="wb") as f:
        pickle.dump(min_params, f)

    clf = lgbm.LGBMRegressor(**min_params)
    clf.fit(X_train, y_train)

    logger.info("train end")

    df = load_test_data()

    X_test = df[np.append(use_cols, "Id")].sort_values("Id")
    X_test.drop(["Id"], axis=1, inplace=True)

    logger.info("test data load end {}".format(X_test.shape))
    pred_test = np.exp(clf.predict(X_test,
                                   num_iteration=clf.best_iteration_))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values("Id")
    df_submit["SalePrice"] = pred_test

    df_submit.to_csv(DIR + "submit.csv", index=False)

    logger.info("end")
