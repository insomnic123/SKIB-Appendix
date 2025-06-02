// Full File Found in Overall Folder
#include <SparkFun_TB6612.h>

#define AIN1 3
#define BIN1 7
#define AIN2 4
#define BIN2 8
#define PWMA 5
#define PWMB 6
#define STBY 9

const int IR_PINS[9] = {22, 23, 24, 25, 26, 27, 28, 29, 30};
const int IR_WEIGHTS[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

float Kp = 10;
float Ki = 0.0;
float Kd = 2 ;

float error = 0;
float lastError = 0;
float integral = 0;
float derivative = 0;

unsigned long lastChangeTime = 0;

Motor motorR = Motor(AIN1, AIN2, PWMA, 1, STBY);
Motor motorL = Motor(BIN1, BIN2, PWMB, 1, STBY);

void setup() {
  Serial.begin(9600);
  for (int pin: IR_PINS) {
    pinMode(pin, INPUT);
  }
}

void loop() {
int activeCount = 0;
int weightedSum = 0;

// Read sensors and calculate weighted sum
for (int i = 0; i < 9; i++) {
  int value = digitalRead(IR_PINS[i]);
  if (value == 1) { // black = line
    weightedSum += IR_WEIGHTS[i];
    activeCount++;
  }
}

float newError = 0;
if (activeCount > 0) {
  newError = (float)weightedSum / activeCount;
} else {
// Line lost — decide a default behavior
  newError = lastError > 0 ? 4 : -4; // Continue in last known direction
}

// Derivative using time
unsigned long now = millis();
unsigned long deltaTime = now - lastChangeTime;

if (newError != lastError && deltaTime > 0) {
  derivative = (newError - lastError) / (float)deltaTime * 1000.0; // per second
  lastChangeTime = now;
} else {
  derivative = 0;
}

error = newError;
integral += error;

float correction = Kp * error + Ki * integral + Kd * derivative;
lastError = error;
Serial.println(correction);

int baseSpeed = 90;
int leftSpeed = baseSpeed + correction;
int rightSpeed = baseSpeed - correction;

leftSpeed = constrain(leftSpeed, 0, 255);
rightSpeed = constrain(rightSpeed, 0, 255);


motorL.drive(leftSpeed);
motorR.drive(rightSpeed);
delay(50);
}