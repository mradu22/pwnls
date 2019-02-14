# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
from keras import backend as K
from datetime import datetime
from sklearn.metrics import matthews_corrcoef
import os
from tqdm import tqdm
import numpy as np


# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        #         score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))
        score = matthews_corrcoef(y_true, (y_proba > threshold).astype(
            np.float64))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, best_score


def gen_name(basename, folder='models/'):
    taken_names = [item for item in os.listdir('models/')]
    base_tag = datetime.now().strftime("%-m%y%d")

    for i in range(65, 91):
        for j in range(65, 91):
            tentative_name = base_tag + chr(i) + chr(j) + '_' + basename
            if not next((s for s in taken_names if tentative_name in s), None):
                return folder + tentative_name


def gen_name_v2(basename, folder='models_v2/'):
    taken_names = [item for item in os.listdir('models_v2/')]
    base_tag = datetime.now().strftime("%-m%y%d")

    for i in range(65, 91):
        for j in range(65, 91):
            tentative_name = base_tag + chr(i) + chr(j) + '_' + basename
            if not next((s for s in taken_names if tentative_name in s), None):
                return folder + tentative_name


def mcc_k(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# this function take a piece of data and convert using transform_ts(), but it does to each of the 3 phases
# if we would try to do in one time, could exceed the RAM Memmory
def prep_data(start, end):
    # load a piece of data from file
    praq_train = pq.read_pandas(
        'input/train.parquet',
        columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    # using tdqm to evaluate processing time
    # takes each index from df_train and iteract it from start to end
    # it is divided by 3 because for each id_measurement there are 3 id_signal, and the start/end parameters are id_signal
    for id_measurement in tqdm(
            df_train.index.levels[0].unique()[int(start / 3):int(end / 3)]):
        X_signal = []
        # for each phase of the signal
        for phase in [0, 1, 2]:
            # extract from df_train both signal_id and target to compose the new data sets
            signal_id, target = df_train.loc[id_measurement].loc[phase]
            # but just append the target one time, to not triplicate it
            if phase == 0:
                y.append(target)
            # extract and transform data into sets of features
            X_signal.append(transform_ts(praq_train[str(signal_id)]))
        # concatenate all the 3 phases in one matrix
        X_signal = np.concatenate(X_signal, axis=1)
        # add the data to X
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
