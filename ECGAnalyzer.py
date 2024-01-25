import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import label
from scipy.stats import mode, zscore

from filters import detrend, butter_filtration, reverse_filt


class ECGAnalyzer:
    def __init__(self, fs=125.0):
        self.fs = fs

        self.raw_data = []
        self.raw_t_sec = []
        self.data = []
        self.t_ms = []

        self.grouped_peaks = []
        self.rr = []
        self.nn = []

        self.filtered = False

    def load_data(self, data, filtered=False, verbose=True):
        """
        :param data: float[]
        :param filtered: bool
        """
        self.filtered = filtered

        self.raw_data = data
        self.raw_t_sec = 1000 * np.arange(len(self.raw_data)) / self.fs

        self.data = data
        self.t_ms = 1000 * np.arange(len(self.data)) / self.fs

        if verbose:
            print("Session length (seconds): " + str(len(self.t_ms) / self.fs))

    def trim_data(self, start, end=0, reset_time=False):
        start_samples = int(start * self.fs)
        end_samples = int(end * self.fs)

        if end_samples == 0:
            end_samples = len(self.data)

        self.data = self.data[start_samples:end_samples]
        if reset_time:
            self.t_ms = 1000 * np.arange(len(self.data)) / self.fs
        else:
            self.t_ms = self.t_ms[start_samples:end_samples]

    def make_filtering(self, cutoff=None, order=2):
        """
        Базовая фильтрация: удаление тренда и фильтрация в две стороны
        для избежания фазовых искажений.
        """
        if cutoff is None:
            cutoff = [25, 0.5]

        if self.filtered:
            raise Exception("Фильтрация уже была проведена")

        filtered_data = detrend(self.data)
        filtered_data = butter_filtration(filtered_data, cutoff=max(cutoff), fs=self.fs, order=order, btype='low')
        filtered_data = reverse_filt(filtered_data, cutoff=max(cutoff), fs=self.fs, order=order, btype='low')
        filtered_data = butter_filtration(filtered_data, cutoff=min(cutoff), fs=self.fs, order=order, btype='high')
        filtered_data = reverse_filt(filtered_data, cutoff=min(cutoff), fs=self.fs, order=order, btype='high')

        self.data = filtered_data
        self.filtered = True

    def calc_rr_and_peaks(self, threshold=0.4, group_threshold=5, corrected=True):
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

        ecg_signal = np.array(self.data)
        ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
        similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
        similarity = similarity / np.max(similarity)

        # raw_peaks = ecg_signal[similarity > threshold].index
        raw_peaks = np.where(similarity > threshold)[0]

        grouped_peaks = np.empty(0)
        peak_groups, num_groups = label(np.diff(raw_peaks) < group_threshold)

        for i in np.unique(peak_groups)[1:]:
            peak_group = raw_peaks[np.where(peak_groups == i)]
            grouped_peaks = np.append(grouped_peaks, np.median(peak_group))

        self.grouped_peaks = grouped_peaks
        self.rr = np.diff(grouped_peaks) * 1000 / self.fs
        if corrected:
            rr_corrected = self.rr.copy()
            rr_corrected[np.abs(zscore(self.rr)) > 2] = np.median(self.rr)
            self.rr = rr_corrected

    def change_fs(self, target_fs):
        if len(self.data) == 0:
            raise Exception('Отсутствуют данные')

        steps = 1 / target_fs

        f = interp1d(self.t_ms, self.data, kind='cubic')
        t_target = np.arange(0, self.t_ms[-1], steps * 1000)

        self.data = f(t_target)
        self.t_ms = t_target
        self.fs = target_fs

    def calc_nn(self, target_fs=0):
        if len(self.rr) == 0:
            raise Exception('Отсутствуют данные по R-R пикам')

        fs = target_fs if target_fs > 0 else self.fs
        steps = 1 / fs

        x = np.cumsum(self.rr) / 1000.0
        f = interp1d(x, self.rr, kind='cubic')

        # now we can sample from interpolation function
        xx = np.arange(1, np.max(x), steps)
        self.nn = f(xx)

    def get_timedomain_metrics(self, normalize=False, verbose=True):
        if len(self.rr) == 0:
            raise Exception('Отсутствуют данные по R-R пикам')

        results = {}

        cardio_intervals = self.nn if normalize else self.rr
        filtered_intervals = [interval for interval in cardio_intervals if 400 <= interval <= 1300]
        cardio_intervals = filtered_intervals

        hr = 60 * 1000 / np.array(cardio_intervals)

        results['Сред. RR (ms)'] = np.mean(cardio_intervals)
        results['std RR (ms)'] = np.std(cardio_intervals)
        results['Сред. ЧСС'] = np.mean(hr)
        results['std ЧСС'] = np.std(hr)
        results['min ЧСС'] = np.min(hr)
        results['max ЧСС'] = np.max(hr)

        # Построение гистограммы
        counts, _ = np.histogram(self.rr, bins=range(400, 1350, 50))

        results['Мода m0'] = mode(cardio_intervals)[0]
        results['Амплитуда am0 (%)'] = max(counts) / len(cardio_intervals) * 100
        results['dRR'] = np.max(cardio_intervals) - np.min(cardio_intervals)
        results['Индекс Баевского ИН'] = results['Амплитуда am0 (%)'] / (2 * results['Мода m0']) * results['dRR']

        # Квадратный корень из среднего значения квадратов последовательных различий между соседними NN
        results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(cardio_intervals))))

        # количество пар последовательных NN, которые отличаются более чем на 50 мс
        results['NN50'] = np.sum(np.abs(np.diff(cardio_intervals)) > 50) * 1

        # доля NN50, деленная на общее количество NN
        results['pNN500 (%)'] = 100 * np.sum((np.abs(np.diff(cardio_intervals)) > 50) * 1) / len(cardio_intervals)

        if verbose:
            for k, v in results.items():
                print("- %s: %.2f" % (k, v))

        return results

    def plot_signal(self, start=0, end=0):
        start_ix = int(start * self.fs)
        end_ix = int(end * self.fs)

        if end == 0:
            end_ix = len(self.data)

        plt.figure(figsize=(20, 7))
        plt.title("ECG signal, slice of %.1f seconds" % ((end_ix - start_ix) / self.fs))
        plt.plot(self.t_ms[start_ix:end_ix], self.data[start_ix:end_ix], color="#51A6D8", linewidth=1)
        plt.xlabel("Time (ms)", fontsize=16)
        plt.ylabel("Amplitude (arbitrary unit)")
        plt.show();

    def plot_rr(self, start=0, end=0):
        if len(self.rr) == 0:
            raise Exception('Отсутствуют данные по R-R пикам')

        start_ix = int(start * self.fs)
        end_ix = int(end * self.fs)

        if end == 0:
            end_ix = len(self.data)

        print("R-R интервал в среднем (мс):", np.mean(self.rr))

        plt.figure(figsize=(20, 7))
        plt.title("Group similar peaks together")
        plt.plot(self.data, label="ECG", color="#51A6D8", linewidth=2)
        plt.plot(
            self.grouped_peaks, np.repeat(1500, self.grouped_peaks.shape[0]), label="median of found peaks", color="k",
            marker="v", linestyle="None"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (arbitrary unit)")
        plt.gca().set_xlim(start_ix, end_ix)
        plt.gca().set_ylim(-1000, 2500)
        plt.show();

    def plot_cardio_histogram(self):
        if len(self.rr) == 0:
            raise Exception('Отсутствуют данные по R-R пикам')

        start = 400
        end = 1300
        step = 50

        # Построение гистограммы
        counts, _, _ = plt.hist(self.rr, bins=range(start, end + step, step))

        mode_value, _ = mode(self.rr)
        print('Мода:', mode_value, max(counts))

        plt.xlabel('Кардиоинтервалы, мс')
        plt.ylabel('Кол-во')
        plt.title('Гистограмма кардиоинтервалов')
        plt.show();
