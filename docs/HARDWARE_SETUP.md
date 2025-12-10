# Murph Hardware Setup Guide

Complete hardware assembly and wiring reference for the Murph desktop robot.

## Table of Contents

- [Bill of Materials](#bill-of-materials)
- [GPIO Pin Assignments](#gpio-pin-assignments)
- [Wiring Diagrams](#wiring-diagrams)
- [Assembly Instructions](#assembly-instructions)
- [Hardware Verification](#hardware-verification)
- [Calibration](#calibration)

---

## Bill of Materials

### Core Components

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Raspberry Pi 5 | Pi 5 4GB/8GB | 1 | 8GB recommended |
| Pi Camera Module 3 | SC0872 | 1 | Wide angle optional |
| MicroSD Card | 32GB+ Class 10 | 1 | A2 rated recommended |
| USB-C Power Supply | 27W 5.1V/5A | 1 | Official Pi 5 PSU |

### Display

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| 1.3" OLED Display | SSD1306 I2C | 1 | 128x64 pixels |

### Motors

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| N20 Micro Gear Motor | 6V 100RPM | 4 | With encoder optional |
| DRV8833 Motor Driver | DRV8833 | 2 | Dual H-bridge |

### Sensors

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| MPU6050 IMU | GY-521 | 1 | 6-axis accel + gyro |
| MPR121 Touch Sensor | Adafruit 1982 | 1 | 12-channel capacitive |

### Audio

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| I2S Microphone | SPH0645 | 1 | MEMS digital mic |
| I2S DAC + Amplifier | MAX98357A | 1 | 3W class D amp |
| Small Speaker | 3W 4Ω | 1 | 28mm diameter |

### Power (Mobile Operation)

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| LiPo Battery | 3.7V 2000mAh | 2 | 18650 or pouch |
| BMS Board | 2S 7.4V 10A | 1 | With balancing |
| DC-DC Converter | 5V 3A Buck | 1 | USB-C output |

### Mechanical

| Component | Quantity | Notes |
|-----------|----------|-------|
| Robot Chassis | 1 | 3D printed or kit |
| Wheels | 4 | 60mm diameter |
| Standoffs M2.5 | 8 | For mounting boards |
| Screws M2.5 | 16 | Various lengths |
| Wires 22AWG | Assorted | Silicone recommended |
| Heat Shrink Tubing | Assorted | For connections |

---

## GPIO Pin Assignments

### Raspberry Pi 5 Pinout

```
                    3.3V [1]  [2]  5V
           I2C SDA (GPIO 2) [3]  [4]  5V
           I2C SCL (GPIO 3) [5]  [6]  GND
                   GPIO 4  [7]  [8]  GPIO 14 (UART TX)
                     GND  [9]  [10] GPIO 15 (UART RX)
                  GPIO 17 [11] [12] GPIO 18 (I2S BCLK)
                  GPIO 27 [13] [14] GND
                  GPIO 22 [15] [16] GPIO 23
                    3.3V [17] [18] GPIO 24
         SPI MOSI (GPIO 10) [19] [20] GND
         SPI MISO (GPIO 9) [21] [22] GPIO 25
         SPI SCLK (GPIO 11) [23] [24] GPIO 8 (SPI CE0)
                     GND [25] [26] GPIO 7 (SPI CE1)
           I2C ID_SD (GPIO 0) [27] [28] GPIO 1 (I2C ID_SC)
                   GPIO 5 [29] [30] GND
                   GPIO 6 [31] [32] GPIO 12 (PWM0)
              PWM1 (GPIO 13) [33] [34] GND
        I2S LRCLK (GPIO 19) [35] [36] GPIO 16
                  GPIO 26 [37] [38] GPIO 20 (I2S DIN)
                     GND [39] [40] GPIO 21 (I2S DOUT)
```

### Pin Assignment Table

| Function | GPIO | Physical Pin | Notes |
|----------|------|--------------|-------|
| **I2C Bus** |
| SDA | GPIO 2 | Pin 3 | Display, IMU, Touch |
| SCL | GPIO 3 | Pin 5 | Display, IMU, Touch |
| **Motor Driver 1 (Front)** |
| AIN1 | GPIO 17 | Pin 11 | Left front direction |
| AIN2 | GPIO 27 | Pin 13 | Left front direction |
| BIN1 | GPIO 22 | Pin 15 | Right front direction |
| BIN2 | GPIO 23 | Pin 16 | Right front direction |
| **Motor Driver 2 (Rear)** |
| AIN1 | GPIO 24 | Pin 18 | Left rear direction |
| AIN2 | GPIO 25 | Pin 22 | Left rear direction |
| BIN1 | GPIO 5 | Pin 29 | Right rear direction |
| BIN2 | GPIO 6 | Pin 31 | Right rear direction |
| **PWM (Motor Speed)** |
| PWM0 | GPIO 12 | Pin 32 | Front motors |
| PWM1 | GPIO 13 | Pin 33 | Rear motors |
| **I2S Audio** |
| BCLK | GPIO 18 | Pin 12 | Bit clock |
| LRCLK | GPIO 19 | Pin 35 | Word select |
| DIN | GPIO 20 | Pin 38 | Data in (from mic) |
| DOUT | GPIO 21 | Pin 40 | Data out (to DAC) |
| **Camera** |
| CSI | - | CSI Port | Pi Camera Module |

### I2C Device Addresses

| Device | Address | Notes |
|--------|---------|-------|
| SSD1306 OLED | 0x3C | Sometimes 0x3D |
| MPU6050 IMU | 0x68 | AD0 to GND |
| MPR121 Touch | 0x5A | Default address |

---

## Wiring Diagrams

### I2C Bus Wiring

All I2C devices share the same bus:

```
Pi GPIO 2 (SDA) ──┬── SSD1306 SDA
                  ├── MPU6050 SDA
                  └── MPR121 SDA

Pi GPIO 3 (SCL) ──┬── SSD1306 SCL
                  ├── MPU6050 SCL
                  └── MPR121 SCL

Pi 3.3V ──────────┬── SSD1306 VCC
                  ├── MPU6050 VCC
                  └── MPR121 VCC

Pi GND ───────────┬── SSD1306 GND
                  ├── MPU6050 GND
                  └── MPR121 GND
```

**Note:** Add 4.7kΩ pull-up resistors on SDA and SCL if not present on breakout boards.

### Motor Driver Wiring (DRV8833)

**Driver 1 (Front Motors):**
```
Pi GPIO 17 ────── AIN1
Pi GPIO 27 ────── AIN2
Pi GPIO 22 ────── BIN1
Pi GPIO 23 ────── BIN2

Pi 5V ─────────── VCC
Pi GND ────────── GND

AOUT1 ─────────── Left Front Motor +
AOUT2 ─────────── Left Front Motor -
BOUT1 ─────────── Right Front Motor +
BOUT2 ─────────── Right Front Motor -
```

**Driver 2 (Rear Motors):**
```
Pi GPIO 24 ────── AIN1
Pi GPIO 25 ────── AIN2
Pi GPIO 5  ────── BIN1
Pi GPIO 6  ────── BIN2

Pi 5V ─────────── VCC
Pi GND ────────── GND

AOUT1 ─────────── Left Rear Motor +
AOUT2 ─────────── Left Rear Motor -
BOUT1 ─────────── Right Rear Motor +
BOUT2 ─────────── Right Rear Motor -
```

### I2S Audio Wiring

**Microphone (SPH0645):**
```
Pi GPIO 18 ────── BCLK
Pi GPIO 19 ────── LRCLK (WS)
Pi GPIO 20 ────── DOUT (Data from mic)
Pi 3.3V ───────── VDD
Pi GND ────────── GND
Pi GND ────────── SEL (Left channel)
```

**DAC/Amplifier (MAX98357A):**
```
Pi GPIO 18 ────── BCLK
Pi GPIO 19 ────── LRC
Pi GPIO 21 ────── DIN (Data to DAC)
Pi 5V ─────────── VIN
Pi GND ────────── GND

Speaker + ─────── +
Speaker - ─────── -
```

### Camera Connection

1. Locate the CSI camera port on Pi 5
2. Lift the plastic clip gently
3. Insert ribbon cable with contacts facing the port
4. Press clip down to secure

### Power Distribution

**For USB-C Powered Operation:**
```
USB-C PSU ─────── Pi 5 USB-C Port
Pi 5V (Pin 2) ──── Motor Drivers, DAC
Pi 3.3V (Pin 1) ── Sensors, Display
```

**For Battery Operation:**
```
LiPo Battery ──── BMS ──── DC-DC Buck (5V) ──── Pi 5 USB-C
                          └── Motor Drivers (VM pin)
```

---

## Assembly Instructions

### 1. Prepare the Chassis

1. 3D print or assemble chassis base
2. Install wheel mounts and motors
3. Route motor wires to center area

### 2. Mount Circuit Boards

1. Install Pi 5 using standoffs (avoid pressure on SD card)
2. Mount motor drivers near motor wires
3. Position IMU near center of mass
4. Mount touch sensor electrodes to shell exterior

### 3. Wire I2C Bus

1. Connect all I2C devices in parallel
2. Use consistent wire colors (red=VCC, black=GND, yellow=SCL, blue=SDA)
3. Keep wires short to minimize interference

### 4. Wire Motors

1. Connect motor driver outputs to motors
2. Test rotation direction before securing
3. Swap motor +/- wires if direction is wrong

### 5. Wire Audio

1. Connect microphone I2S lines
2. Connect DAC I2S lines (shares BCLK/LRCLK with mic)
3. Secure speaker in enclosure

### 6. Install Camera

1. Connect ribbon cable to Pi CSI port
2. Mount camera with clear field of view
3. Avoid ribbon cable stress

### 7. Final Assembly

1. Secure all wires with zip ties
2. Install battery (if mobile)
3. Close enclosure

---

## Hardware Verification

### 1. Power Test

```bash
# Check Pi power
vcgencmd measure_volts core
vcgencmd measure_temp

# Should see ~1.0V core, <60°C temp
```

### 2. I2C Scan

```bash
sudo i2cdetect -y 1
```

Expected output:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c -- -- --
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- 5a -- -- -- -- --
60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```

Addresses: 0x3c (OLED), 0x5a (MPR121), 0x68 (MPU6050)

### 3. Camera Test

```bash
# List cameras
libcamera-hello --list-cameras

# Preview
libcamera-hello -t 5000

# Capture image
libcamera-still -o test.jpg
```

### 4. Audio Test

```bash
# List audio devices
arecord -l
aplay -l

# Test recording (5 seconds)
arecord -D plughw:0,0 -f S16_LE -r 16000 -d 5 test.wav

# Play back
aplay test.wav
```

### 5. Motor Test

```python
# Quick motor test script
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)

# Spin left front motor forward
GPIO.output(17, GPIO.HIGH)
GPIO.output(27, GPIO.LOW)
time.sleep(1)

# Stop
GPIO.output(17, GPIO.LOW)
GPIO.output(27, GPIO.LOW)

GPIO.cleanup()
```

### 6. IMU Test

```python
import smbus

bus = smbus.SMBus(1)
address = 0x68

# Wake up MPU6050
bus.write_byte_data(address, 0x6B, 0)

# Read accelerometer
accel_x = bus.read_byte_data(address, 0x3B)
print(f"Accel X raw: {accel_x}")
```

---

## Calibration

### IMU Calibration

The MPU6050 needs calibration for accurate readings:

1. Place robot on level surface
2. Run calibration routine to measure bias
3. Store offsets for runtime correction

Expected at rest:
- Accel X, Y: ~0g
- Accel Z: ~-1g (gravity)
- Gyro X, Y, Z: ~0 deg/s

### Touch Sensor Calibration

MPR121 auto-calibrates, but you can tune sensitivity:

1. Adjust baseline filter (0x2B-0x2E registers)
2. Set touch/release thresholds (0x41-0x5A)
3. Test with intended electrode materials

### Motor Calibration

For straight-line driving:

1. Measure actual speed at various PWM values
2. Calculate left/right motor trim
3. Adjust in `shared/constants.py` or runtime config

### Face Distance Calibration

The `VISION_FACE_DISTANCE_CALIBRATION_PX` constant (160) assumes a face height of 160 pixels at 50cm distance. Adjust based on your camera:

1. Place face at known distance (50cm)
2. Measure detected face height in pixels
3. Update constant proportionally
