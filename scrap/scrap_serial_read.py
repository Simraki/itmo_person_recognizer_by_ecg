import serial
from matplotlib import pyplot as plt

# Инициализация серийного порта
ser = serial.Serial('COM2', 9600)  # Указать нужный COM-порт и скорость передачи данных

# Чтение данных с порта и построение графика
data = []
while True:
    # Wait until there is data waiting in the serial buffer
    line = ser.readline().decode().strip()  # Чтение строки с порта и декодирование из байтов

    if line == 'end':
        print('close socket')
        break

    if not line.lstrip('-+').replace('.', '', 1).isdigit():
        continue

    data.append(float(line))
    plt.plot(data)  # Построение графика
    plt.pause(0.01)  # Задержка между обновлениями графика

# Закрытие серийного порта
ser.close()
