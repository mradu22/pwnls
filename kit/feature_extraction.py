import numpy as np
from dask import dataframe as dd
from dask.multiprocessing import get
from dask import delayed
from dask import compute
from multiprocessing import cpu_count
from dask.distributed import Client
from scipy import signal as sgn

nCores = cpu_count()
# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128

M = 3000
rate = 1 / (2e-2 / 800000)


# This function standardize the data from (-128 to 127) to (-1 to 1)
# Theoretically it helps in the NN Model training, but I didn't tested without it
def min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (
            range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


# This is one of the most important peace of code of this Kernel
# Any power line contain 3 phases of 800000 measurements, or 2.4 millions data
# It would be praticaly impossible to build a NN with an input of that size
# The ideia here is to reduce it each phase to a matrix of <n_dim> bins by n features
# Each bin is a set of 5000 measurements (800000 / 160), so the features are extracted from this 5000 chunk data.
def transform_ts(ts,
                 n_dim=500,
                 min_max=(-1, 1),
                 sample_size=800000,
                 rescale_=False):
    # convert data into -1 to 1
    if rescale_:
        ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    else:
        ts_std = ts
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std()  # standard deviation
        std_top = mean + std  # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 3, 25, 50, 75, 97, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean  # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        new_ts.append(
            np.concatenate([
                np.asarray([mean, std, max_range, std_top, std_bot]),
                percentil_calc, relative_percentile
            ]))
    return np.asarray(new_ts)


def transform_ts_spectrogram(ts,
                             n_dim=299,
                             min_max=(-1, 1),
                             sample_size=800000):
    _, _, Sx = sgn.spectrogram(
        ts.values,
        fs=rate,
        window='hanning',
        nperseg=M,
        noverlap=None,
        detrend='constant',
        scaling='spectrum')
    Sxp = np.mean(Sx[:1500, :].reshape(30, 50, 304), axis=1).T

    return np.log10(Sxp)


def prep_data(df_meta, signal_data, train=True):
    X_temp = []

    if train:
        y = []
        for id_measurement in df_meta.index.levels[0].unique():
            for phase in [0, 1, 2]:
                signal_id, target = df_meta.loc[id_measurement].loc[phase]

                if phase == 0:
                    y.append(target)

                ss = delayed(transform_ts_spectrogram)(
                    signal_data[str(signal_id)])
                X_temp.append(ss)

        X_temp = compute(X_temp, scheduler='processes')[0]
        X = np.asarray([
            np.concatenate(X_temp[i:i + 3], axis=1)
            for i in range(0, len(X_temp), 3)
        ])
        y = np.asarray(y)
        return X, y
    else:
        for id_measurement in df_meta.index.levels[0].unique():
            for phase in [0, 1, 2]:
                signal_id = int(df_meta.loc[id_measurement].loc[phase])

                ss = delayed(transform_ts_spectrogram)(
                    signal_data[str(signal_id)])
                X_temp.append(ss)

        X_temp = compute(X_temp, scheduler='processes')[0]
        X = np.asarray([
            np.concatenate(X_temp[i:i + 3], axis=1)
            for i in range(0, len(X_temp), 3)
        ])
        return X
