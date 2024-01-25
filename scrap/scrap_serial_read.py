import serial
from matplotlib import pyplot as plt

ser = serial.Serial('COM2', 9600)

data = []
while True:
    # print(ser.readline().decode(encoding='ascii', errors="ignore").strip())
    line = ser.readline().decode().strip()  # Чтение строки с порта и декодирование из байтов

    if line == 'end':
        print('close socket')
        break

    if not line.lstrip('-+').replace('.', '', 1).isdigit():
        continue

    data.append(float(line))
    # plt.plot(data)
    # plt.pause(0.01)

ser.close()
