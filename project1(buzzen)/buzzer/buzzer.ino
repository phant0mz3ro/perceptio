const int trigPin = 9;
const int echoPin = 10;
const int buzzer = 8;

long duration;
int distance;

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzer, OUTPUT);
}

void loop() {
  // 🔊 Handle buzzer command from Python
  if (Serial.available() > 0) {
    char data = Serial.read();

    if (data == '1') {
      digitalWrite(buzzer, HIGH);
    }
    else if (data == '0') {
      digitalWrite(buzzer, LOW);
    }
  }

  // 📏 Ultrasonic measurement
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;

  // 📡 Send distance to Python
  Serial.println(distance);

  delay(100);
}