#include <Arduino.h>
#include "BluetoothSerial.h"
#include <Adafruit_NeoPixel.h>

BluetoothSerial SerialBT;

// --- [핀 설정] ---
const int PRESSURE_PIN   = 34;   // 압력센서 (ADC1)
const int PELTIER_PIN    = 18;   // 릴레이 IN1 → 펠티어 + 쿨링팬 (같이 켜짐)
const int HUMIDIFIER_PIN = 19;   // 가습기 (2N2222 베이스로)
const int LED_PIN        = 23;   // WS2812B
// const int BUTTON_PIN  = 27;   // [테스트 중 주석] ON/OFF 버튼
const int NUM_PIXELS     = 8;

// 릴레이 모듈이 LOW에서 켜지면 true, HIGH에서 켜지면 false
const bool RELAY_ACTIVE_LOW = true;

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

// [테스트 중] 버튼 없이 항상 ON. 버튼을 다시 쓸 때는 false로 바꾸고 아래 버튼 코드 주석을 푸세요.
bool systemOn  = true;

// Jetson(또는 시리얼 모니터)에서 받은 상태
bool jetsonWarn   = false;
bool jetsonDanger = false;

// 최종 상태 (Jetson 상태 + 핸들 이탈을 합친 결과)
bool isWarning = false;
bool isDanger  = false;

// --- 음성 알림: 부저 대신 Jetson에 "!BUZ#" 전송 ---
const unsigned long BUZ_REPEAT_MS = 3000;  // 경고가 계속되는 동안 다시 보내는 간격 (0이면 시작할 때 한 번만)
int lastAlertLevel = 0;                    // 0: 정상, 1: 경고, 2: 위험
unsigned long lastBuzSent = 0;

unsigned long lastPressureTime = 0;
const unsigned long PRESSURE_OFF_DELAY = 1000;  // 핸들에서 손을 뗀 뒤 이 시간이 지나면 위험
const int PRESSURE_THRESHOLD = 400;

// --- [테스트 중 주석] 버튼 디바운스 ---
// int lastButtonReading = HIGH;
// int buttonState = HIGH;
// unsigned long lastDebounceTime = 0;
// const unsigned long DEBOUNCE_MS = 40;

void peltier(bool on) {
  digitalWrite(PELTIER_PIN, (on == RELAY_ACTIVE_LOW) ? LOW : HIGH);
}

void sendBuz() {
  Serial.print("!BUZ#\n");
  if (SerialBT.hasClient()) SerialBT.print("!BUZ#\n");
  lastBuzSent = millis();
}

void stopAllAlerts() {
  jetsonWarn = false;
  jetsonDanger = false;
  isWarning = false;
  isDanger = false;
  lastAlertLevel = 0;
  peltier(false);
  digitalWrite(HUMIDIFIER_PIN, LOW);
  strip.clear();
  strip.show();
}

// [테스트 중 주석] 버튼으로 시스템 켜고 끄기
// void setSystem(bool on) {
//   systemOn = on;
//   stopAllAlerts();
//   if (on) lastPressureTime = millis();   // 켜는 순간 핸들 이탈로 오판하지 않도록
//   Serial.println(on ? "[버튼] 시스템 ON" : "[버튼] 시스템 OFF");
// }

// void handleButton() {
//   int reading = digitalRead(BUTTON_PIN);
//   if (reading != lastButtonReading) lastDebounceTime = millis();
//   if (millis() - lastDebounceTime > DEBOUNCE_MS && reading != buttonState) {
//     buttonState = reading;
//     if (buttonState == LOW) setSystem(!systemOn);   // 눌렀을 때 토글
//   }
//   lastButtonReading = reading;
// }

void setup() {
  Serial.begin(115200);

  pinMode(PRESSURE_PIN, INPUT);
  // pinMode(BUTTON_PIN, INPUT_PULLUP);   // [테스트 중 주석]

  peltier(false);                       // 릴레이가 순간적으로 붙지 않도록 먼저 OFF 값을 쓰고 출력 설정
  pinMode(PELTIER_PIN, OUTPUT);
  peltier(false);
  pinMode(HUMIDIFIER_PIN, OUTPUT);
  digitalWrite(HUMIDIFIER_PIN, LOW);

  strip.begin();
  strip.setBrightness(80);   // 테스트용: 밝기를 낮춰 5V 전류를 줄임 (최대 255)
  strip.clear();
  strip.show();

  SerialBT.begin("Jetson_ESP32_BT");
  lastPressureTime = millis();

  Serial.println("======================================");
  Serial.println("ESP32: 압력센서·가습기·LED·펠티어+팬, 경고 시 !BUZ# 전송 (버튼 없이 항상 ON)");
  Serial.println("명령: !LV1_WARN#  !LV2_DANGER#  !OFF#  !H#");
  Serial.println("======================================");
}

void processCommand(String rawData, Stream &client) {
  rawData.trim();
  if (rawData.startsWith("!") && rawData.indexOf("#") != -1) {
    // OFF 상태: 통신 응답(A)은 하되 동작 명령은 무시
    if (!systemOn) {
      client.print("A\n");
      return;
    }
    if (rawData.indexOf("LV1_WARN") != -1) {
      jetsonWarn = true; jetsonDanger = false;
      client.print("A\n");
      Serial.println("[상태 수신] LV1_WARN");
    } else if (rawData.indexOf("LV2_DANGER") != -1) {
      jetsonWarn = false; jetsonDanger = true;
      client.print("A\n");
      Serial.println("[상태 수신] LV2_DANGER");
    } else if (rawData.indexOf("OFF") != -1) {
      stopAllAlerts();
      client.print("A\n");
      Serial.println("[상태 수신] OFF");
    } else if (rawData.indexOf("H") != -1) {
      client.print("A\n");
    }
  }
}

void loop() {
  // handleButton();   // [테스트 중 주석]

  if (Serial.available() > 0) {
    String usbData = Serial.readStringUntil('\n');
    processCommand(usbData, Serial);
  }
  if (SerialBT.available() > 0) {
    String btData = SerialBT.readStringUntil('\n');
    processCommand(btData, SerialBT);
    while (SerialBT.available() > 0) SerialBT.read();
  }

  if (!systemOn) {          // OFF: 센서·명령 모두 무시, 출력은 전부 꺼진 상태 유지
    delay(20);
    return;
  }

  // --- 압력센서 / 핸들 이탈 판정 ---
  int pressureVal = analogRead(PRESSURE_PIN);
  if (pressureVal >= PRESSURE_THRESHOLD) lastPressureTime = millis();
  bool handsOff = (millis() - lastPressureTime >= PRESSURE_OFF_DELAY);

  // Jetson 상태와 핸들 이탈을 합쳐서 최종 상태 결정
  isDanger  = jetsonDanger || handsOff;
  isWarning = jetsonWarn && !isDanger;

  // --- 음성 알림 (부저 대신) ---
  // 경고/위험이 새로 시작되거나 경고→위험으로 올라가면 바로 보내고,
  // 계속되는 동안 BUZ_REPEAT_MS마다 다시 보냄
  int alertLevel = isDanger ? 2 : (isWarning ? 1 : 0);
  if (alertLevel > 0) {
    bool escalated = alertLevel > lastAlertLevel;
    bool repeatDue = BUZ_REPEAT_MS > 0 && millis() - lastBuzSent >= BUZ_REPEAT_MS;
    if (escalated || repeatDue) sendBuz();
  }
  lastAlertLevel = alertLevel;

  // --- 액추에이터 제어 ---
  if (isWarning) {
    peltier(false);
    digitalWrite(HUMIDIFIER_PIN, LOW);
    for (int i = 0; i < strip.numPixels(); i++) strip.setPixelColor(i, strip.Color(255, 180, 0));
    strip.show();
  } else if (isDanger) {
    peltier(true);
    digitalWrite(HUMIDIFIER_PIN, HIGH);
    for (int i = 0; i < strip.numPixels(); i++) strip.setPixelColor(i, strip.Color(255, 0, 0));
    strip.show();
  } else {
    peltier(false);
    digitalWrite(HUMIDIFIER_PIN, LOW);
    for (int i = 0; i < strip.numPixels(); i++) strip.setPixelColor(i, strip.Color(0, 255, 0));
    strip.show();
  }

  delay(20);
}
