import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import serial
from PyQt5 import QtSerialPort, QtCore
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QApplication, QHBoxLayout, QStyle, \
    QFileDialog, QMessageBox, QLineEdit, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar

from ECGAnalyzer import ECGAnalyzer
from ECGIndentifier import ECGIndentifier

matplotlib.use('Qt5Agg')

ecg_identifier = ECGIndentifier(preload=True)

target_analyzer: ECGAnalyzer = None


def process_data(data):
    analyzer = ECGAnalyzer(fs=125.0)

    analyzer.load_data(data=data, verbose=False)
    analyzer.trim_data(30, 150, reset_time=True)
    analyzer.make_filtering()
    analyzer.calc_rr_and_peaks(threshold=0.3)
    metrics = analyzer.get_timedomain_metrics(verbose=False)

    return [*metrics.values()], analyzer


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        super(Canvas, self).__init__(fig)
        self.setParent(parent)

    def clear(self):
        self.figure.clear()

    def plot_signal(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(target_analyzer.t_ms / 1000, target_analyzer.data, color="#51A6D8", linewidth=1)
        ax.set_xlabel("Time (sec)", fontsize=16)
        ax.set_ylabel("Amplitude (arbitrary unit)")
        ax.set_title("ECG signal")
        self.draw()

    def plot_cardio_histogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        metrics = target_analyzer.get_timedomain_metrics(verbose=False)

        counts, _, _ = ax.hist(target_analyzer.rr, bins=range(400, 1350, 50))

        mode_value = metrics['Мода m0']
        ax.axvline(x=int(mode_value), color='r', linestyle='--')

        ax.set_xlabel('Cardiointervals, ms')
        ax.set_ylabel("Counts")
        ax.set_title("Cardiointervals histogram")
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        title = "ECG Identifier 007"
        self.online_mode = False
        self.online_data = []
        self.online_analyzer: ECGAnalyzer = None

        self.setWindowTitle(title)

        # ==========================

        self.new_person_label_text_input = QLineEdit(self)
        self.new_person_label_text_input.textChanged.connect(
            self.check_label_for_new_person
        )

        # ==========================

        self.button_add_new_person = QPushButton(
            icon=self.style().standardIcon(QStyle.SP_FileDialogStart), text="Add new person by ECG", parent=self
        )
        self.button_add_new_person.setEnabled(False)
        self.button_add_new_person.clicked.connect(self.add_new_person)

        # ==========================

        self.button_show_storage_info = QPushButton("Show storage info", self)
        self.button_show_storage_info.clicked.connect(self.show_storage_info)

        # ==========================

        self.button_load_person_for_check = QPushButton(
            icon=self.style().standardIcon(QStyle.SP_FileDialogStart),
            text="Load person for checking",
            parent=self
        )
        self.button_load_person_for_check.clicked.connect(self.load_person_for_check)

        # ==========================

        self.button_toggle_online_person_check = QPushButton(
            text="Start online checking",
            parent=self
        )
        self.button_toggle_online_person_check.toggled.connect(self.toggle_online_person_check)

        # ==========================

        self.button_check_person = QPushButton("Check person", self)
        self.button_check_person.setEnabled(False)
        self.button_check_person.clicked.connect(self.predict_person)

        # ==========================

        self.button_show_plots = QPushButton("Show plots", self)
        self.button_show_plots.setEnabled(False)
        self.button_show_plots.clicked.connect(self.show_plots)

        # ==========================

        self.button_show_timedomain_metrics = QPushButton("Show timedomain metrics", self)
        self.button_show_timedomain_metrics.setEnabled(False)
        self.button_show_timedomain_metrics.clicked.connect(self.show_timedomain_metrics)

        # ==========================

        self.add_person_btn_layout = QHBoxLayout()
        self.add_person_btn_layout.addWidget(self.new_person_label_text_input)
        self.add_person_btn_layout.addWidget(self.button_add_new_person)
        self.add_person_btn_layout.addWidget(self.button_show_storage_info)
        self.add_person_btn_layout.setStretch(0, 2)
        self.add_person_btn_layout.setStretch(1, 1)
        self.add_person_btn_layout.setStretch(2, 1)
        self.add_person_btn_widget = QWidget()
        self.add_person_btn_widget.setLayout(self.add_person_btn_layout)

        # ==========================

        self.load_btn_layout = QHBoxLayout()
        self.load_btn_layout.addWidget(self.button_load_person_for_check)
        self.load_btn_layout.addWidget(self.button_toggle_online_person_check)
        self.load_btn_widget = QWidget()
        self.load_btn_widget.setLayout(self.load_btn_layout)

        # ==========================

        self.check_btn_layout = QHBoxLayout()
        self.check_btn_layout.addWidget(self.button_check_person)
        self.check_btn_layout.addWidget(self.button_show_plots)
        self.check_btn_layout.addWidget(self.button_show_timedomain_metrics)
        self.check_btn_widget = QWidget()
        self.check_btn_widget.setLayout(self.check_btn_layout)

        # ==========================

        self.filename_label = QLabel(self)
        self.result_label = QLabel(self)

        self.labels_layout = QHBoxLayout()
        self.labels_layout.addWidget(self.filename_label)
        self.labels_layout.addWidget(self.result_label)
        self.labels_widget = QWidget()
        self.labels_widget.setLayout(self.labels_layout)

        # ==========================

        self.signal_plotter = Canvas(self, width=4, height=2)
        self.cardio_hist_plotter = Canvas(self, width=6, height=3)

        signal_toolbar = NavigationToolbar(self.signal_plotter, self)
        cardio_hist_toolbar = NavigationToolbar(self.cardio_hist_plotter, self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.add_person_btn_widget)
        self.layout.addWidget(self.load_btn_widget)
        self.layout.addWidget(self.check_btn_widget)

        self.layout.addWidget(self.labels_widget)

        self.layout.addWidget(signal_toolbar)
        self.layout.addWidget(self.signal_plotter)

        self.layout.addWidget(cardio_hist_toolbar)
        self.layout.addWidget(self.cardio_hist_plotter)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.setCentralWidget(self.widget)
        self.show()

        self.serial = QtSerialPort.QSerialPort(
            'COM2',
            baudRate=QtSerialPort.QSerialPort.Baud9600,
            readyRead=self.receive_online_data
        )

    @QtCore.pyqtSlot()
    def receive_online_data(self):
        while self.serial.canReadLine():
            line = self.serial.readLine().data().decode().strip()

            if not line.lstrip('-+').replace('.', '', 1).isdigit():
                continue

            self.online_data.append(float(line))
            self.update_all_by_online_data(with_predict=len(self.online_data) > 4 * 125.0)

    def update_all_by_online_data(self, with_predict=False):
        global target_analyzer
        analyzer = ECGAnalyzer(fs=125.0)
        analyzer.load_data(self.online_data)
        analyzer.make_filtering()
        analyzer.calc_rr_and_peaks(threshold=0.3)

        target_analyzer = analyzer

        self.show_plots()
        if with_predict:
            self.predict_person()

    def check_label_for_new_person(self):
        if self.new_person_label_text_input.text():
            self.button_add_new_person.setEnabled(True)
        else:
            self.button_add_new_person.setEnabled(False)

    def add_new_person(self):
        label = self.new_person_label_text_input.text()
        if not label:
            QMessageBox.critical(self, "Error", "Provide label for new person")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select CSV File with ECG Column", filter='*.csv')
        if file_path:
            try:
                temp_df = pd.read_csv(file_path)
                temp_data = temp_df['ECG'].to_numpy()
                data = process_data(temp_data)[0]
                ecg_identifier.add_train_data(data, label)
                ecg_identifier.calc_pca_knn()

                info = ecg_identifier.get_storage_info()
                n_persons = len(info.keys())
                n_records = sum(info.values())

                QMessageBox.information(
                    self, "Success",
                    f"Data of new person has been uploaded. Database has {n_persons} persons and {n_records} records"
                )
            except Exception as e:
                self.button_check_person.setEnabled(False)
                self.button_show_plots.setEnabled(False)
                self.button_show_timedomain_metrics.setEnabled(False)
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")

    @QtCore.pyqtSlot(bool)
    def toggle_online_person_check(self, checked):
        print(checked)
        # self.online_mode = not self.online_mode
        # self.button_load_person_for_check.setEnabled(not self.online_mode)
        # self.button_check_person.setEnabled(not self.online_mode)
        # self.button_show_plots.setEnabled(not self.online_mode)
        # self.button_show_timedomain_metrics.setEnabled(not self.online_mode)
        #
        # self.button_toggle_online_person_check.setText(
        #     'Stop online checking' if self.online_mode else 'Start online checking'
        # )
        #
        # if self.online_mode:
        #     self.signal_plotter.clear()
        #     self.cardio_hist_plotter.clear()
        #
        #     ser = serial.Serial('COM2', 9600)  # Указать нужный COM-порт и скорость передачи данных
        #
        #     while True:
        #         # Wait until there is data waiting in the serial buffer
        #         line = ser.readline().decode().strip()  # Чтение строки с порта и декодирование из байтов
        #
        #         if line == 'end':
        #             print('close socket')
        #             break
        #
        #         if not line.lstrip('-+').replace('.', '', 1).isdigit():
        #             continue
        #
        #         data.append(float(line))
        #         plt.plot(data)  # Построение графика
        #         plt.pause(0.01)  # Задержка между обновлениями графика

    def load_person_for_check(self):
        global target_analyzer
        self.filename_label.setText('')
        self.result_label.setText('')
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select CSV File with ECG Column", filter='*.csv')
        if file_path:
            try:
                path, filename = os.path.split(file_path)
                self.filename_label.setText(f"File uploaded: {filename}")
                temp_df = pd.read_csv(file_path)
                temp_data = temp_df['ECG'].to_numpy()
                target_analyzer = process_data(temp_data)[1]
                self.button_check_person.setEnabled(True)
                self.button_show_plots.setEnabled(True)
                self.button_show_timedomain_metrics.setEnabled(True)
            except Exception as e:
                self.button_check_person.setEnabled(False)
                self.button_show_plots.setEnabled(False)
                self.button_show_timedomain_metrics.setEnabled(False)
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")

    def predict_person(self):
        metrics = target_analyzer.get_timedomain_metrics(verbose=False)
        d = [*metrics.values()]
        predicted_label = ecg_identifier.predict_target_data(d)
        self.result_label.setText(
            f"Predicted person: {predicted_label}"
        )

    def show_plots(self):
        self.signal_plotter.plot_signal()
        self.cardio_hist_plotter.plot_cardio_histogram()

    def show_timedomain_metrics(self):
        output_str = ""
        for key, value in target_analyzer.get_timedomain_metrics(verbose=False).items():
            output_str += f"{key}: {round(value, 2)} \n\n"
        QMessageBox.information(self, "Timedomain metrics", output_str)

    def show_storage_info(self):
        output_str = "Class => records"
        for key, value in ecg_identifier.get_storage_info().items():
            output_str += f"{key} => {value} \n"
        QMessageBox.information(self, "Storage info", output_str)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
