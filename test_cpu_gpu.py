import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq  # Used to read the data
import os
import numpy as np
from keras.layers import *  # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model
from tqdm import tqdm  # Processing time measurement
from sklearn.model_selection import train_test_split
from keras import backend as K  # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers  # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Used to use Kfold to train our model
from keras.callbacks import *  # This object helps the model to train in a smarter way, avoiding overfitting
from kit.attention import Attention
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from kit.utils import mcc_k
from numba import jit
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import time
from dask import dataframe as dd
from dask.multiprocessing import get
from dask import delayed
from dask import compute
from multiprocessing import cpu_count
from dask.distributed import Client
nCores = cpu_count()
import tensorflow as tf
from keras.utils import multi_gpu_model

from kit.genesis import FeatureExtractor, data_prep
from kit.feature_extraction import prep_data
from kit.utils import gen_name, threshold_search, gen_name_v2

import os
##TRAIN ONLY ON GPU-0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

X = np.load("datasets/X_train.npy")
y = np.load("datasets/y_train.npy")

print(X.shape)
# This is NN LSTM Model creation
# The X shape here is very important. It is also important undertand a little how a LSTM works
# X.shape[0] is the number of id_measuremts contained in train data
# X.shape[1] is the number of chunks resultant of the transformation, each of this date enters in the LSTM serialized
# This way the LSTM can understand the position of a data relative with other and activate a signal that needs
# a serie of inputs in a specifc order.
# X.shape[3] is the number of features multiplied by the number of phases (3)

feature_config = {
    'time_steps': 400,
    'spectrogram_freq_bins': 30,
    'abs_rescale': 0,
    'stats_mean': 0,
    'stats_std': 0,
    'stats_std_top': 0,
    'stats_std_bot': 0,
    'stats_max_range': 0,
    'stats_percentiles': 0,
    'stats_relative_percentiles': 0
}

model_config = {
    'hu1': 256,
    'hu2': 128,
    'dr1': 0.3,
    'dr2': 0.4,
    'de1': 64,
    'parallel': False
}

train_config = {'val_split': 0.2, 'stages_desc': [(4096, 2)]}


def model_one(input_shape, kw):
    inp = Input(shape=(
        input_shape[1],
        input_shape[2],
    ))
    x = Bidirectional(LSTM(kw['hu1'], return_sequences=True))(inp)
    x = Dropout(rate=kw['dr1'])(x)
    x = Bidirectional(LSTM(kw['hu2'], return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(kw['de1'], activation="relu")(x)
    x = Dropout(rate=kw['dr2'])(x)
    x = Dense(1, activation="sigmoid")(x)
    if kw['parallel']:
        model = Model(inputs=inp, outputs=x)
        parallel_model = multi_gpu_model(model)
        parallel_model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=[mcc_k])

        return parallel_model
    else:
        model = Model(inputs=inp, outputs=x)
        model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=[mcc_k])

        return model


def train_model_base(X, y, tcf, model=model_one, model_config=model_config):

    val_split = tcf['val_split']
    stages_desc = tcf['stages_desc']

    t0 = time.time()
    stages = dict.fromkeys([i for i in range(len(stages_desc))])

    K.clear_session()
    model = model_one(X.shape, model_config)
    model.summary()

    with K.tf.device('/cpu:0'):
        config = tf.ConfigProto(
            log_device_placement=True, device_count={
                'GPU': 0,
                'CPU': 1
            })
        session = tf.Session(config=config)
        K.set_session(session)

    for i in range(len(stages_desc)):
        stages[i] = {
            'batch_size': stages_desc[i][0],
            'epochs': stages_desc[i][1]
        }
        hist_temp = model.fit(
            X,
            y,
            batch_size=stages_desc[i][0],
            epochs=stages_desc[i][1],
            validation_split=val_split)
        stages[i]['results'] = hist_temp.history

    model_results = {'config': model.to_json(), 'training': stages}

    best_score = max(model_results['training'][0]['results']['val_mcc_k'])
    print('=' * 100 + '\n' + '=' * 100)
    elapsed_time = round(time.time() - t0, 3) / 60.0
    print('Time elapsed: {} minutes... Best score: {} '.format(
        elapsed_time, best_score))

    return model_results, elapsed_time


out_ = train_model_base(
    X, y, train_config, model=model_one, model_config=model_config)
