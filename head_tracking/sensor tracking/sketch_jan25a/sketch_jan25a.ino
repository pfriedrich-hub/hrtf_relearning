int analogPin = A0;     //initializing pin 2 as ‘pwm’ variable
const int min_input = 0;
const int max_input = 180;
const int min_output = 0;
const int max_output = 1023;
float y = 0;
float output = 0;
int angle = 0;
int count = 0;

void setup()
{
     analogWriteResolution(10);
     Serial.begin(115200);
     // SYNTAX
     //pinMode(analogPin, OUTPUT);
}

void loop()
{
  y = 90;
  //count ++;
  //y = ((sin(angle) + 1)/2) * 360;
  //y = 180 + (10 * count) % 180;
  //y = random(min_input, max_input);
  // y = 180;
  output = map(y, min_input, max_input, min_output, max_output);
  analogWrite(analogPin, output);
    // sets DAC pin to an output voltage of 1024/4095 * 3.3V = 0.825V.
  
  delay(100) ;
  Serial.println(y);
  Serial.println(output);
  angle++;
}
