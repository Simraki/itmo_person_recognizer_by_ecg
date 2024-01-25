import time

import pandas as pd
import serial

fs = 125.0

df = pd.read_csv('test/VadimMonoECG.csv')
data = df['ECG'].to_numpy()

data = data[:10]

# Инициализация серийного порта
ser = serial.Serial('COM1', 9600)  # Указать нужный COM-порт и скорость передачи данных

# Отправка данных на порт
for d in data:
    out = f"{d}\n".encode()  # Преобразование строки в байты
    ser.write(out)
    print(d)
    time.sleep(1 / fs)  # Задержка между отправками записей

print('end')
# ser.write('end\n'.encode())

# Закрытие серийного порта
ser.close()
