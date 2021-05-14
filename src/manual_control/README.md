# arduino_nano_v3
Neccessary codes and libraries for the ardunio nano on the model car(version 3.0).
Copy the libraries (MPU6050, I2Cdev,Adafruit_NeoPixel) to the arduino-1.6.9/libraries folder.

    /opt/arduino-1.8.5/hardware/tools/avr/bin/avrdude -C/opt/arduino-1.8.5/hardware/tools/avr/etc/avrdude.conf -v -patmega328p -carduino -P/dev/ttyArduino -b57600 -D -Uflash:w:/root/IMU_Zero.ino.eightanaloginputs.hex:i


    /opt/arduino-1.8.5/hardware/tools/avr/bin/avrdude -C/opt/arduino-1.8.5/hardware/tools/avr/etc/avrdude.conf -v -patmega328p -carduino -P/dev/ttyArduino -b57600 -D -Uflash:w:/root/main.ino.eightanaloginputs.hex:i

