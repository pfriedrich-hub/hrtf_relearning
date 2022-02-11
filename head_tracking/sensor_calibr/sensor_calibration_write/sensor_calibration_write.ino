#include "Adafruit_Sensor_Calibration.h"

// select either EEPROM or SPI FLASH storage:
#ifdef ADAFRUIT_SENSOR_CALIBRATION_USE_EEPROM
  Adafruit_Sensor_Calibration_EEPROM cal;
#else
  Adafruit_Sensor_Calibration_SDFat cal;
#endif

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  delay(100);
  Serial.println("Calibration filesys test");
  if (!cal.begin()) {
    Serial.println("Failed to initialize calibration helper");
    while (1) yield();
  }
  Serial.print("Has EEPROM: "); Serial.println(cal.hasEEPROM());
  Serial.print("Has FLASH: "); Serial.println(cal.hasFLASH());

  if (! cal.loadCalibration()) {
    Serial.println("No calibration loaded/found... will start with defaults");
  } else {
    Serial.println("Loaded existing calibration");
  }

// in uTesla
  cal.mag_hardiron[0] = -58.12;
  cal.mag_hardiron[1] = 6.4;
  cal.mag_hardiron[2] = 1.435;

  // in uTesla
  cal.mag_softiron[0] = 1.094;
  cal.mag_softiron[1] = 0.037;
  cal.mag_softiron[2] = 0.011;  
  cal.mag_softiron[3] = 0.037;
  cal.mag_softiron[4] = 0.945;
  cal.mag_softiron[5] = 0.008;  
  cal.mag_softiron[6] = 0.011;
  cal.mag_softiron[7] = 0.008;
  cal.mag_softiron[8] = 0.969;
  // Earth total magnetic field strength in uTesla (dependent on location and time of the year),
  // visit: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm)
  cal.mag_field = 35.51; // approximate value for locations along the equator

  // in Radians/s
  cal.gyro_zerorate[0] = 0.0623;
  cal.gyro_zerorate[1] = -0.1191;
  cal.gyro_zerorate[2] = -0.0776; 

  if (! cal.saveCalibration()) {
    Serial.println("**WARNING** Couldn't save calibration");
  } else {
    Serial.println("Wrote calibration");    
  }

  cal.printSavedCalibration();
}

void loop() {

}
