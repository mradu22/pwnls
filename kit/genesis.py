import numpy as np
from scipy import signal as sgn
import numpy as np
from dask import dataframe as dd
from dask.multiprocessing import get
from dask import delayed
from dask import compute
from multiprocessing import cpu_count
from dask.distributed import Client
from scipy import signal as sgn


class FeatureExtractor:
    def __init__(self, param_config, N=800000):
        """Note1: if using spectrogram the numer of time steps
        will be approximate.
        """
        self.N = N
        self.rate = 1 / (2e-2 / self.N)
        self.param_config = param_config
        self.spec_per_seg = 8 * self.N // (7 * self.param_config['time_steps'])

        self.flist = {
            'stats_mean':
            np.mean,
            'stats_std':
            np.std,
            'stats_std_top':
            lambda x: np.std(x) + np.mean(x),
            'stats_std_bot':
            lambda x: np.mean(x) - np.std(x),
            'stats_max_range':
            lambda x: np.max(x) - np.min(x),
            'stats_percentiles':
            np.percentile,
            'stats_relative_percentiles':
            lambda x, y: np.percentile(x, y) - np.mean(x)
        }

        assert set(self.flist.keys()) == set(
            [item for item in param_config.keys() if 'stats_' in item])

    def extract_all(self, data):
        d_master = []

        if self.param_config['spectrogram_freq_bins'] != 0:
            d_master.append(self.generate_spectrogram(data))

        if self.param_config['abs_rescale'] == 1:
            d_master.append(
                self.generates_stat_features(self.absolute_rescale(data)))
        else:
            d_master.append(self.generates_stat_features(data))

        if self.param_config['spectrogram_freq_bins'] == 0:
            return d_master[0]
        else:
            BULLSHIT = min([i.shape[0] for i in d_master])
            return np.concatenate(
                [d_master[0][:BULLSHIT, :], d_master[1][:BULLSHIT, :]], axis=1)

    def generates_stat_features(self, data):
        bucket_size = int(self.N / (self.param_config['time_steps'] - 1))
        new_ts = []
        for i in range(0, self.N, bucket_size):
            data_chunk = data[i:i + bucket_size]
            temp = []
            for k, v in self.param_config.items():
                if "stats_" in k:
                    if v == 1:
                        temp.append(self.flist[k](data_chunk))
                    elif v != 0:
                        temp += list(self.flist[k](data_chunk, v))
            new_ts.append(np.asarray(temp))
        return np.asarray(new_ts)

    def absolute_rescale(self,
                         ts,
                         min_data=-128,
                         max_data=127,
                         range_needed=(-1, 1)):
        ##rescales dataset based on absolute values of min and max
        ##maintains relational information between datapoints
        if min_data < 0:
            ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
        else:
            ts_std = (ts - min_data) / (max_data - min_data)
        if range_needed[0] < 0:
            return ts_std * (
                range_needed[1] + abs(range_needed[0])) + range_needed[0]
        else:
            return ts_std * (
                range_needed[1] - range_needed[0]) + range_needed[0]

    def local_rescale(self, data, min, max):
        pass
        ##rescales data based on relative values of min and max
        ##note: this loses relational information between datapoints

    def generate_spectrogram(self, signal):
        _, _, Sxp = sgn.spectrogram(
            signal,
            fs=self.rate,
            window='hanning',
            nperseg=self.spec_per_seg,
            noverlap=None,
            detrend='constant',
            scaling='spectrum')

        temp_bin = Sxp.shape[0] // self.param_config['spectrogram_freq_bins']
        #maybe can be removed later to make faster

        Sx_condensed = np.mean(
            Sxp[:(self.param_config['spectrogram_freq_bins'] * temp_bin
                  ), :].reshape(self.param_config['spectrogram_freq_bins'],
                                temp_bin, Sxp.shape[1]),
            axis=1).T

        self.param_config['time_steps'] = Sxp.shape[1]

        return np.log10(Sx_condensed)


def data_prep(df_meta, signal_data, generator, train=True):
    X_temp = []

    if train:
        y = []
        for id_measurement in df_meta.index.levels[0].unique():
            for phase in [0, 1, 2]:
                signal_id, target = df_meta.loc[id_measurement].loc[phase]

                if phase == 0:
                    y.append(target)

                ss = delayed(generator.extract_all)(
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

                ss = delayed(generator.extract_all)(
                    signal_data[str(signal_id)])
                X_temp.append(ss)

        X_temp = compute(X_temp, scheduler='processes')[0]
        X = np.asarray([
            np.concatenate(X_temp[i:i + 3], axis=1)
            for i in range(0, len(X_temp), 3)
        ])
        return X
