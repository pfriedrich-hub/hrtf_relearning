// Full orientation sensing using NXP/Madgwick/Mahony and a range of 9-DoF
// sensor sets.
// You *must* perform a magnetic calibration before this code will work.
//
// To view this data, use the Arduino Serial Monitor to watch the
// scrolling angles, or run the OrientationVisualiser example in Processing.
// Based on  https://github.com/PaulStoffregen/NXPMotionSense with adjustments
// to Adafruit Unified Sensor interface

#include <Adafruit_Sensor_Calibration.h>
#include <Adafruit_AHRS.h>

Adafruit_Sensor *accelerometer, *gyroscope, *magnetometer;

// uncomment one combo 9-DoF!
#include "LSM6DS_LIS3MDL.h"  // can adjust to LSM6DS33, LSM6DS3U, LSM6DSOX...
//#include "LSM9DS.h"           // LSM9DS1 or LSM9DS0
//#include "NXP_FXOS_FXAS.h"  // NXP 9-DoF breakout

// pick your filter! slower == better quality output
//Adafruit_NXPSensorFusion filter; // slowest
//Adafruit_Madgwick filter;  // faster than NXP
Adafruit_Mahony filter;  // fastest/smalleset

#if defined(ADAFRUIT_SENSOR_CALIBRATION_USE_EEPROM)
  Adafruit_Sensor_Calibration_EEPROM cal;
#else
  Adafruit_Sensor_Calibration_SDFat cal;
#endif

#define FILTER_UPDATE_RATE_HZ 100
#define PRINT_EVERY_N_UPDATES 10
//#define AHRS_DEBUG_OUTPUT

uint32_t timestamp;

int analogPinAz = DAC0;     //initializing pin 
int analogPinEle = DAC1;     //initializing pin 

const int az_min_input = 0;
const int az_max_input = 360;
const int ele_min_input = -90;
const int ele_max_input = 90;

const int min_output = 0;
const int max_output = 4095;

float output = 0;
float az = 0;
float ele = 0;

void setup() {
//  analogWriteResolution(10);
  Serial.begin(115200);
  while (!Serial) yield();
  //if (!cal.begin()) {
  //  Serial.println("Failed to initialize calibration helper");
  //} else if (! cal.loadCalibration()) {
  //  Serial.println("No calibration loaded/found");
  //}
  
  cal.mag_hardiron[0] = -48.83; // works pretty well
  cal.mag_hardiron[1] = 13.77;  
  cal.mag_hardiron[2] = -4.84;
  cal.mag_softiron[0] = 1.150;
  cal.mag_softiron[1] = 0.194;
  cal.mag_softiron[2] = 0.137;  
  cal.mag_softiron[3] = 0.194;
  cal.mag_softiron[4] = 0.991;
  cal.mag_softiron[5] = 0.000;  
  cal.mag_softiron[6] = 0.137;
  cal.mag_softiron[7] = 0.000;
  cal.mag_softiron[8] = 0.924;  
  cal.mag_field = 40.47; 


  if (!init_sensors()) {
    Serial.println("Failed to find sensors");
    while (1) delay(10);
  }
  
  //accelerometer->printSensorDetails();
  //gyroscope->printSensorDetails();
  //magnetometer->printSensorDetails();

  setup_sensors();
  filter.begin(FILTER_UPDATE_RATE_HZ);
  timestamp = millis();

  //Wire.setClock(400000); // 400KHz
}


void loop() {

  float roll, pitch, heading;
  float gx, gy, gz;
  static uint8_t counter = 0;

  if ((millis() - timestamp) < (1000 / FILTER_UPDATE_RATE_HZ)) {
    return;
  }

  timestamp = millis();
  // Read the motion sensors
  sensors_event_t accel, gyro, mag;
  accelerometer->getEvent(&accel);
  gyroscope->getEvent(&gyro);
  magnetometer->getEvent(&mag);

  cal.calibrate(mag);
  cal.calibrate(accel);
  cal.calibrate(gyro);
  // Gyroscope needs to be converted from Rad/s to Degree/s
  // the rest are not unit-important
  gx = gyro.gyro.x * SENSORS_RADS_TO_DPS;
  gy = gyro.gyro.y * SENSORS_RADS_TO_DPS;
  gz = gyro.gyro.z * SENSORS_RADS_TO_DPS;
   
  // Update the SensorFusion filter
  filter.update(gx, gy, gz, 
                accel.acceleration.x, accel.acceleration.y, accel.acceleration.z, 
                mag.magnetic.x, mag.magnetic.y, mag.magnetic.z);

  // only print the calculated output once in a while
  if (counter++ <= PRINT_EVERY_N_UPDATES) {
    return;
  }
  // reset the counter
  counter = 0;

#if defined(AHRS_DEBUG_OUTPUT)
  //Serial.print("Raw: ");
  Serial.print(accel.acceleration.x, 4); Serial.print(", ");
  Serial.print(accel.acceleration.y, 4); Serial.print(", ");
  Serial.print(accel.acceleration.z, 4); Serial.print(", ");
  Serial.print(gx, 4); Serial.print(", ");
  Serial.print(gy, 4); Serial.print(", ");
  Serial.print(gz, 4); Serial.print(", ");
  Serial.print(mag.magnetic.x, 4); Serial.print(", ");
  Serial.print(mag.magnetic.y, 4); Serial.print(", ");
  Serial.print(mag.magnetic.z, 4); Serial.println("");
#endif

  // print the heading, pitch and roll
  roll = -filter.getRoll();
  pitch = filter.getPitch();
  heading = 360-filter.getYaw();
  //Serial.print("Orientation: ");
  Serial.print("az: "); 
  Serial.print(heading);
  Serial.print(", ele: ");
  Serial.println(roll);
  //Serial.print(pitch);
  //Serial.println(", ");

  // write to analog pin
  analogWriteResolution(12);
  az = map(heading, az_min_input, az_max_input, min_output, max_output);
  analogWrite(analogPinAz,az) ;     //setting  
  ele = map(roll, ele_min_input, ele_max_input, min_output, max_output);
  analogWrite(analogPinEle,ele) ;
  Serial.print("az out: "); 
  Serial.print(az);
  Serial.print(", ele out: ");
  Serial.println(ele);
  
#if defined(AHRS_DEBUG_OUTPUT)
 // Serial.print("Took "); Serial.print(millis()-timestamp); Serial.println(" ms");
#endif
}
