from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from ECGAnalyzer import ECGAnalyzer


def preload_data():
    def get_ecg_data(filepath):
        ecg_df = pd.read_csv(filepath)
        return ecg_df['ECG'].to_numpy()

    def process(ecg_data):
        analyzer = ECGAnalyzer(fs=125.0)

        analyzer.load_data(data=ecg_data, verbose=False)
        analyzer.trim_data(30, 150, reset_time=True)
        analyzer.make_filtering()
        analyzer.calc_rr_and_peaks(threshold=0.3)
        metrics = analyzer.get_timedomain_metrics(verbose=False)

        return analyzer.data, [*metrics.values()]

    def get_trim_data(data_dict):
        data = []
        for data_key in data_dict.keys():
            temp = process(data_dict[data_key])
            data.append(temp[1])
        X = np.array(data)
        return X

    test_vadim_stress = get_ecg_data('test/VadimStressECG.csv')
    test_vadim_mono = get_ecg_data('test/VadimMonoECG.csv')
    test_vlad_mono = get_ecg_data('test/VladMonoECG.csv')
    test_ilya_stress = get_ecg_data('test/IlyaStressECG.csv')
    test_ilya_mono = get_ecg_data('test/IlyaMonoECG.csv')
    test_adelya_mono = get_ecg_data('for_add/AdelyaMonoECG.csv')
    test_adelya_stress = get_ecg_data('for_add/AdelyaStressECG.csv')

    tests = {
        'test_vadim_stress': test_vadim_stress,
        'test_vadim_mono': test_vadim_mono,
        'test_vlad_mono': test_vlad_mono,
        'test_ilya_stress': test_ilya_stress,
        'test_ilya_mono': test_ilya_mono,
        # 'test_adelya_mono': test_adelya_mono,
        # 'test_adelya_stress': test_adelya_stress
    }

    # y_test = ['vadim', 'vadim', 'vlad', 'ilya', 'ilya', 'adelya', 'adelya']
    y_test = ['vadim', 'vadim', 'vlad', 'ilya', 'ilya']
    X_test = get_trim_data(tests)

    return X_test, y_test


class ECGIndentifier:
    def __init__(self, preload=False):
        self.labels = []

        self.train_data = []
        self.train_labels = []

        self.trust_coeff = 0

        self.le = LabelEncoder()
        self.pca = None
        self.knn = None

        if preload:
            X_test, y_test = preload_data()
            self.train_data = np.array(X_test)

            self.labels = [x for x in y_test]
            self.train_labels = self.le.fit_transform(self.labels)

            self.calc_pca_knn()

    def set_train_data_with_labels(self, labeled_train_data):
        """
            :param labeled_train_data: [label: str, data: float[]]
        """
        self.train_data = np.array([x[1] for x in labeled_train_data])

        self.labels = [x[0] for x in labeled_train_data]
        self.train_labels = self.le.fit_transform(self.labels)

    def add_train_data(self, train_data, label):
        if len(train_data) != len(self.train_data[0]):
            raise Exception(f"Check train_data lengths: {len(train_data)} (must be {len(self.train_data[0])})")
        self.train_data = np.vstack((self.train_data, train_data))
        self.labels.append(label)
        self.le = LabelEncoder()
        self.train_labels = self.le.fit_transform(self.labels)

    def calc_pca_knn(self, n_neighbors=1):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.train_data)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_pca, self.train_labels)

        y_pred = knn.predict(X_pca)
        accuracy = accuracy_score(self.train_labels, y_pred)

        self.pca = pca
        self.knn = knn
        self.trust_coeff = accuracy

    def predict_target_data(self, target_data):
        X_pca = self.pca.transform([target_data])
        y_pred = self.knn.predict(X_pca)
        return self.le.inverse_transform(y_pred)[0]

    def get_storage_info(self):
        counter = Counter(self.labels)
        return OrderedDict(counter.most_common())
