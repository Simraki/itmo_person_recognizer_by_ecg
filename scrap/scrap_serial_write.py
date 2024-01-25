import time

import pandas as pd
import serial

fs = 125.0

df = pd.read_csv('test/VadimMonoECG.csv')
data = df['ECG'].to_numpy()

data = data[:10]

ser = serial.Serial('COM1', 9600)

for d in data:
    out = f"{d}\n".encode()
    ser.write(out)
    print(d)
    time.sleep(1 / fs)

# print('end')
# ser.write('end\n'.encode())

ser.close()
