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
import time, re, pickle
from dask import dataframe as dd
from dask.multiprocessing import get
from dask import delayed
from dask import compute
from multiprocessing import cpu_count
from dask.distributed import Client
nCores = cpu_count()
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import ParameterGrid
from keras.utils import multi_gpu_model
from keras.models import model_from_json

from kit.genesis import FeatureExtractor, data_prep
from kit.utils import gen_name, threshold_search, gen_name_v2

#
# feature_config = {
#     'time_steps': 300,
#     'spectrogram_freq_bins': 50,
#     'abs_rescale': 0,
#     'stats_mean': 0,
#     'stats_std': 0,
#     'stats_std_top': 0,
#     'stats_std_bot': 0,
#     'stats_max_range': 0,
#     'stats_percentiles': 0,
#     'stats_relative_percentiles': 0
# }
#
# model_config = {
#     'hu1': 128,
#     'hu2': 64,
#     'dr1': 0.3,
#     'dr2': 0.4,
#     'de1': 64,
#     'parallel': True
# }
#
# train_config = {'val_split': 0.2, 'stages_desc': [(500, 2)]}


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


def train_model_cv(X, y, full_config, N_SPLITS=5):
    t0 = time.time()

    stages_desc = full_config['train_config']['stages_desc']
    model_name = re.findall("/([a-zA-Z0-9]*)", full_config['file_name'])[0]

    model = model_one(X.shape, full_config['model_config'])
    model.summary()

    # with K.tf.device('/cpu:0'):
    #     config = tf.ConfigProto(
    #         log_device_placement=True, device_count={
    #             'GPU': 0,
    #             'CPU': 1
    #         })
    #     session = tf.Session(config=config)
    #     K.set_session(session)

    splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                        random_state=1985).split(X, y))
    preds_val = []
    y_val = []

    cvmodels = dict.fromkeys(
        ['split_' + str(i) for i in range(1, N_SPLITS + 1)])

    for idx, (train_idx, val_idx) in enumerate(splits):
        K.clear_session()
        stages = dict.fromkeys([i for i in range(len(stages_desc))])
        print("Beginning fold {}".format(idx + 1))
        train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[
            val_idx], y[val_idx]
        model = model_one(X.shape, full_config['model_config'])
        ckpt = ModelCheckpoint(
            'models/' + model_name + '_weight' + '_{}.h5'.format(idx),
            save_best_only=True,
            save_weights_only=True,
            verbose=False,
            monitor='val_mcc_k',
            mode='max')

        for i in range(len(stages_desc)):
            stages[i] = {
                'batch_size': stages_desc[i][0],
                'epochs': stages_desc[i][1]
            }
            hist_temp = model.fit(
                train_X,
                train_y,
                batch_size=stages_desc[i][0],
                epochs=stages_desc[i][1],
                validation_data=[val_X, val_y],
                callbacks=[ckpt])
            stages[i]['results'] = hist_temp.history

        cvmodels['split_' + str(idx + 1)] = stages

        preds_val.append(model.predict(val_X, batch_size=512))
        y_val.append(val_y)

    # concatenates all and prints the shape
    preds_val = np.concatenate(preds_val)[..., 0]
    y_val = np.concatenate(y_val)

    preds_val.shape, y_val.shape
    best_threshold, best_score = threshold_search(y_val, preds_val)

    model_results = {
        'file_name': model_name,
        'training': cvmodels,
        'metrics': {
            'roc_curve': roc_curve(y_val, preds_val),
            'roc_auc_score': roc_auc_score(y_val, preds_val),
            'best_threshold': best_threshold,
            'best_mcc': best_score
        }
    }

    with open('models/' + model_name + '.pkl', 'wb') as fp:
        pickle.dump(model_results, fp)

    print('=' * 80)
    print('=' * 80)
    print('Best threshold: {}\nBest score: {}\nTime elapsed: {} minutes...'.
          format(best_threshold, best_score,
                 round(time.time() - t0, 4) / 60.0))
    print('=' * 80)
    print('=' * 80)


def get_full_config(model_name):
    la = [item for item in os.listdir('models_v2/') if 'pkl' in item]
    list_of_models = list(
        set([
            laitem for item in [model_name] for laitem in la if item in laitem
        ]))[0]
    with open('models_v2/' + list_of_models, 'rb') as fp:
        return pickle.load(fp)


if __name__ == '__main__':
    model_name = '21909BA'
    full_config = get_full_config(model_name)

    t0 = time.time()

    print('Loading data...')
    meta_train, df_train = load_data()
    print(full_config['feature_config'])
    print('Extracting features...')
    feature_pipe = FeatureExtractor(param_config=full_config['feature_config'])
    X, y = data_prep(meta_train, df_train, feature_pipe)
    print(
        'done. It took {} minutes.'.format((time.time() - t0) / 60.0), end='')
    print('Dataset shape: {}'.format(X.shape))

    train_model_cv(X, y, full_config, N_SPLITS=5)
