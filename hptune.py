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
import matplotlib.pyplot as plt
from kit.utils import mcc_k
from numba import jit
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
from sklearn.model_selection import ParameterGrid
from keras.utils import multi_gpu_model

from kit.genesis import FeatureExtractor, data_prep
from kit.utils import gen_name, threshold_search, gen_name_v2

##################################################
##############TRAIN ONLY ON GPU-0#################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##################################################
##################################################

feature_config = {
    'time_steps': 200,
    'spectrogram_freq_bins': 30,
    'abs_rescale': 1,
    'stats_mean': 0,
    'stats_std': 0,
    'stats_std_top': 0,
    'stats_std_bot': 0,
    'stats_max_range': 0,
    'stats_percentiles': 0,
    'stats_relative_percentiles': 0
}

model_config = {
    'hu1': 128,
    'hu2': 64,
    'dr1': 0.3,
    'dr2': 0.4,
    'de1': 64,
    'parallel': True
}

train_config = {'val_split': 0.2, 'stages_desc': [(128, 10), (800, 50)]}


def check_hardware():
    list_of_gpus = K.tensorflow_backend._get_available_gpus()
    no_gpu = len([item for item in list_of_gpus if 'GPU' in item])
    if no_gpu > 0:
        print('Found {} GPUs.'.format(no_gpu))
    else:
        print('No GPU found!!')

    print('Found {} CPU cores.'.format(nCores))


def load_data():
    meta_train = pd.read_csv('input/metadata_train.csv')
    meta_train = meta_train.set_index(['id_measurement', 'phase'])
    df_train = pq.read_pandas('input/train.parquet').to_pandas()
    return meta_train, df_train


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

    # with K.tf.device('/gpu:0'):
    #     config = tf.ConfigProto(
    #         log_device_placement=True, device_count={
    #             'GPU': 4,
    #             'CPU': 1
    #         })
    #     session = tf.Session(config=config)
    #     K.set_session(session)

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


def hptuner(feature_config,
            model_config,
            train_config,
            data_hyperparam_list=None,
            model_hyperparam_list=None):

    check_hardware()
    meta_train, df_train = load_data()

    if (not data_hyperparam_list) and (not model_hyperparam_list):
        print('No parameters are being varied.')
        return
    if not data_hyperparam_list:
        data_hyperparam_list = [{'1': 1}]
    if not model_hyperparam_list:
        model_hyperparam_list = [{'1': 1}]

    for data_hyperparameter_set in data_hyperparam_list:
        feature_config.update(data_hyperparameter_set)
        print('Extracting features...')

        for v, k in feature_config.items():
            print(v + ':' + str(k))

        feature_pipe = FeatureExtractor(param_config=feature_config)
        X, y = data_prep(meta_train, df_train, feature_pipe)
        print('done.', end='')
        # X = np.load('datasets/X_train.npy')
        # y = np.load('datasets/y_train.npy')
        print(X.shape, y.shape)

        for hyperparameter_set in model_hyperparam_list:

            model_config.update(hyperparameter_set)
            train_config.update(hyperparameter_set)
            for v, k in model_config.items():
                print(v + ':' + str(k))
            for v, k in train_config.items():
                print(v + ':' + str(k))

            t0 = time.time()
            try:
                out_, elapsed_time = train_model_base(
                    X,
                    y,
                    train_config,
                    model=model_one,
                    model_config=model_config)
                out_['elapsed_time'] = elapsed_time
                out_['model_config'] = model_config
                out_['train_config'] = train_config
                out_['feature_config'] = feature_config

            except:
                out_ = {}
                out_['training'] = 'FAILED'
                out_['elapsed_time'] = elapsed_time
                out_['model_config'] = model_config
                out_['train_config'] = train_config

            file_name = gen_name_v2('hpt')
            out_['file_name'] = file_name
            with open(file_name + '.pkl', 'wb') as fp:
                pickle.dump(out_, fp)


if __name__ == '__main__':
    data_grid = {
        'spectrogram_freq_bins': [30, 0],
        'stats_mean': [0, 1],
        'stats_std': [0, 1],
        'stats_std_top': [0, 1],
        'stats_std_bot': [0, 1],
        'stats_max_range': [0, 1],
        'stats_percentiles': [0, [0, 10, 25, 50, 75, 90, 100]],
        'stats_relative_percentiles': [0, [0, 10, 25, 50, 75, 90, 100]]
    }
    param_grid = {}

    param_grid = list(ParameterGrid(param_grid))
    data_grid = list(ParameterGrid(data_grid))

    print('{} total models to run.'.format(len(param_grid) + len(data_grid)))

    hptuner(
        feature_config,
        model_config,
        train_config,
        model_hyperparam_list=param_grid,
        data_hyperparam_list=data_grid)
