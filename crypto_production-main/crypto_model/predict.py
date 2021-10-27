import os
import joblib
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import lru_cache
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

from config import SCALER, FEATURES, MODEL, NUM_FEATURES
from utils import reshape_as_image
import warnings

warnings.filterwarnings('ignore')


def main(df):
    # Load Scaler & transform

    temp = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'index_close']].copy()
    scaler = get_scaler()
    df[FEATURES] = scaler.transform(df[FEATURES])

    # Load Model
    model = get_model()
    output = _predict(model, df)

    output = output.sort_values(by=['per_prob_score']).tail(3)
    output = output.merge(temp, on=['date', 'ticker'], how='left')
    return output


def _predict(model, df):
    x_test = df[FEATURES].values
    dim = int(np.sqrt(NUM_FEATURES))
    x_test = reshape_as_image(x_test, dim, dim)
    x_test = np.stack((x_test,) * 3, axis=-1)
    pred_prob = model.predict(x_test)
    df['Probability'] = pred_prob[:, 1]
    prob_data = df[['date', 'ticker', 'Probability']].copy()
    prob_data['per_prob_score'] = prob_data.groupby(['date'])['Probability'].rank(pct=True)
    return prob_data


@lru_cache(maxsize=5)
def get_scaler():
    return joblib.load(SCALER)


@lru_cache(maxsize=5)
def get_model():
    return load_model(MODEL)


if __name__ == "__main__":
    df = pd.read_csv('../data/all_features.csv')
    prob_df = main(df)
    prob_df.to_csv('../output/historical_predictions.csv', index=False)
    print(prob_df.tail())