import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ECGAnalyzer import ECGAnalyzer


def get_ecg_data(filepath):
    ecg_df = pd.read_csv(filepath)
    return ecg_df['ECG'].to_numpy()


def process(ecg_data, end=180):
    analyzer = ECGAnalyzer(fs=125.0)

    analyzer.load_data(data=ecg_data, verbose=False)
    analyzer.trim_data(30, end, reset_time=True)
    analyzer.make_filtering()
    analyzer.calc_rr_and_peaks(threshold=0.3)
    metrics = analyzer.get_timedomain_metrics(verbose=False)
    baes = metrics['Индекс Баевского ИН']

    return analyzer.data, analyzer.raw_data, analyzer.rr, baes, metrics.values()


def get_trim_data(data_dict, mode=0):
    min_size = None
    data = []
    for data_key in data_dict.keys():
        temp = process(data_dict[data_key])
        if min_size is None or min_size > len(temp[mode]):
            min_size = len(temp[mode])
        data.append([temp[mode], temp[4]])

    X = []
    for d in data:
        trim_d = d[0][:min_size]
        X.append([*d[1]])
    X = np.array(X)
    return X


test_vadim_stress = get_ecg_data('test/VadimStressECG.csv')
test_vadim_mono = get_ecg_data('test/VadimMonoECG.csv')
test_vlad_mono = get_ecg_data('test/VladMonoECG.csv')
test_vlad_stress = get_ecg_data('test/VladStressECG.csv')
test_ilya_stress = get_ecg_data('test/IlyaStressECG.csv')
test_ilya_mono = get_ecg_data('test/IlyaMonoECG.csv')

val_vadim_stress = get_ecg_data('val/VadimStressECG.csv')
val_vadim_mono = get_ecg_data('val/VadimMonoECG.csv')
val_vlad_mono = get_ecg_data('val/VladMonoECG.csv')
val_vlad_stress = get_ecg_data('val/VladStressECG.csv')
val_ilya_stress = get_ecg_data('val/IlyaStressECG.csv')
val_ilya_mono = get_ecg_data('val/IlyaMonoECG.csv')

tests = {
    'test_vadim_stress': test_vadim_stress,
    'test_vadim_mono': test_vadim_mono,
    'test_vlad_mono': test_vlad_mono,
    # 'test_vlad_stress': test_vlad_stress,
    'test_ilya_stress': test_ilya_stress,
    'test_ilya_mono': test_ilya_mono
}

vals = {
    'val_ilya_stress': val_ilya_stress,
    'val_ilya_mono': val_ilya_mono,
    'val_vadim_mono': val_vadim_mono,
    'val_vlad_mono': val_vlad_mono,
    'val_vlad_stress': val_vlad_stress,
    'val_vadim_stress': val_vadim_stress,
}

# y_test = np.array([0, 0, 2, 2, 1, 1])
y_test = np.array([0, 0, 2, 1, 1])
y_val = np.array([1, 1, 0, 2, 2, 0])

X_test = get_trim_data(tests)
X_val = get_trim_data(vals)


def test_classifier(classifier, y_test, y_val):
    pca = PCA(n_components=3)
    X_test_pca = pca.fit_transform(X_test)
    classifier.fit(X_test_pca, y_test)

    y_pred = classifier.predict(X_test_pca)
    print(y_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Точность модели (test): {:.2f}%".format(accuracy * 100))

    X_val_pca = pca.transform(X_val)
    y_pred = classifier.predict(X_val_pca)
    print(y_val)
    print(y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    print("Точность модели (val): {:.2f}%".format(accuracy * 100))


test_classifier(KNeighborsClassifier(n_neighbors=1), y_test, y_val)
