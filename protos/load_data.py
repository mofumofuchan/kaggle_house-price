#!/usr/bin/env python3
# coding: utf-8
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from logging import (getLogger, Formatter, StreamHandler,
                     DEBUG, FileHandler)

TRAIN_DATA = "../input/train.csv"
TEST_DATA = "../input/test.csv"

DIR = "result_tmp/"
LOG_FILENAME = "train_xgb.py.log"


logger = getLogger(__name__)
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


def read_csv(path):
    logger.info("enter")
    df = pd.read_csv(path)

    for i in tqdm(range(df.shape[1])):
        if df.iloc[:, i].dtypes == object:
            labeler = LabelEncoder()
            labeler.fit(list(df.iloc[:, i].values))
            df.iloc[:, i] = labeler.transform(list(df.iloc[:, i].values))

    logger.info("exit")

    return df


def load_train_data():
    logger.info("enter")
    df = read_csv(TRAIN_DATA)
    logger.info("exit")

    return df


def load_test_data():
    logger.info("enter")
    df = read_csv(TEST_DATA)
    logger.info("exit")

    return df


if __name__ == "__main__":
    print(load_train_data().head())
    print(load_test_data().head())
