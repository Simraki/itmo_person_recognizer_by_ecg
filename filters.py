import numpy as np
from scipy import signal


def detrend(data):
    data = np.array(data)
    data = data.astype('float32')
    data -= np.mean(data)
    return data


def butter_filter_coeff(cutoff, fs, order, btype):
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_filtration(data, cutoff, fs, order, btype):
    b, a = butter_filter_coeff(cutoff, fs, order, btype)
    return signal.lfilter(b, a, data)


def reverse_filt(data, cutoff, fs, order, btype):
    reverse_data = data[::-1]
    b, a = butter_filter_coeff(cutoff, fs, order, btype)
    return signal.lfilter(b, a, reverse_data)[::-1]
