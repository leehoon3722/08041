import serial
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)

ser.write(b"TEST\n")

response = ser.readline()
print(f"결과: {response.decode()}")