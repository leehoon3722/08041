import serial
import time

ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=1)

print("--- 통신 테스트 시작 ---")

try:
    while True:
        print("Sending: H")
        ser.write(b'H') 

        response = ser.read(1).decode('utf-8')

        if response == 'A':
            print("✅ Received: A (통신 성공!)")
        else:
            print(f"❌ No Response (받은 데이터: {response})")

        time.sleep(1) 

except KeyboardInterrupt:
    print("\n통신 테스트를 종료합니다.")
    ser.close()
