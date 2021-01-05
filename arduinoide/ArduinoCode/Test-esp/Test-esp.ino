void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.flush();
  Serial.begin(250000); // Baud-rate
  Serial.print("Test");
}
