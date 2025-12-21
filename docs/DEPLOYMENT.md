# Murph Deployment Guide

Step-by-step deployment instructions for server, Raspberry Pi, and emulator.

## Table of Contents

- [Requirements](#requirements)
- [Server Deployment](#server-deployment)
- [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [Emulator Deployment](#emulator-deployment)
- [Systemd Services](#systemd-services)
- [Network Configuration](#network-configuration)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Server Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Ubuntu 22.04 LTS, Debian 12, macOS 13+ | Ubuntu 24.04 LTS |
| **Python** | 3.11 | 3.11 or 3.12 |
| **RAM** | 4GB | 8GB+ (for Ollama) |
| **Storage** | 2GB | 10GB+ (for LLM models) |
| **Network** | LAN access to Pi | Gigabit Ethernet |

**Supported Operating Systems:**
- Ubuntu 22.04 LTS / 24.04 LTS
- Debian 12 (Bookworm)
- macOS 13 Ventura / 14 Sonoma / 15 Sequoia
- Windows 11 with WSL2 (Ubuntu)
- Fedora 39/40 (community tested)

### Raspberry Pi Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Hardware** | Raspberry Pi 5 (4GB) | Raspberry Pi 5 (8GB) |
| **OS** | Raspberry Pi OS Bookworm (64-bit) | Latest Raspberry Pi OS |
| **Storage** | 16GB microSD | 32GB+ A2-rated microSD |
| **Power** | 5V/3A USB-C | Official 27W Pi 5 PSU |

**Required Pi OS Version:**
- Raspberry Pi OS (64-bit) based on Debian 12 Bookworm
- Kernel 6.1+ (for Pi 5 support)
- Released: October 2023 or later

**Required Hardware Interfaces:**
- Camera (CSI) - enabled via raspi-config
- I2C - enabled via raspi-config
- I2S - enabled via /boot/firmware/config.txt
- SPI - optional, enable if needed

---

## Server Deployment

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# For microphone support (emulator/testing):
sudo apt install -y libportaudio2 portaudio19-dev
```

**macOS:**
```bash
brew install python@3.11 git portaudio
```

### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add to PATH (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Clone and Install

```bash
git clone git@github.com:YourBr0ther/murph.git
cd murph
poetry install
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

Key settings:
- `MURPH_LLM_PROVIDER` - Set to `ollama`, `nanogpt`, or `mock`
- `NANOGPT_API_KEY` - Required if using NanoGPT

### 5. Install Ollama (Optional)

If using Ollama for local LLM:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull llama3.2-vision

# Start Ollama service
ollama serve
```

### 6. Run Server

```bash
poetry run python -m server.main
```

Dashboard available at: `http://localhost:6081`

---

## Raspberry Pi Deployment

### 1. Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Select: Raspberry Pi OS (64-bit)
3. Configure: hostname, SSH, WiFi, locale
4. Flash to SD card

### 2. Enable Hardware Interfaces

```bash
sudo raspi-config
```

Enable:
- Interface Options → Camera → Enable
- Interface Options → I2C → Enable
- Interface Options → SPI → Enable (if needed)

For I2S audio, edit `/boot/firmware/config.txt`:
```ini
# Enable I2S
dtparam=i2s=on

# I2S microphone
dtoverlay=googlevoicehat-soundcard

# Or for generic I2S DAC:
# dtoverlay=hifiberry-dac
```

Reboot after changes:
```bash
sudo reboot
```

### 3. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libcamera-dev \
    libcamera-apps \
    python3-libcamera \
    python3-picamera2 \
    libportaudio2 \
    portaudio19-dev \
    libatlas-base-dev \
    i2c-tools
```

### 4. Verify Hardware

```bash
# Check I2C devices
sudo i2cdetect -y 1

# Expected addresses:
# 0x3C - SSD1306 OLED Display
# 0x5A - MPR121 Touch Sensor
# 0x68 - MPU6050 IMU

# Test camera
libcamera-hello --list-cameras
```

### 5. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 6. Clone and Install

```bash
git clone git@github.com:YourBr0ther/murph.git
cd murph
poetry install
```

**Note:** Pi hardware libraries (picamera2, RPi.GPIO, smbus2, luma.oled) are installed via apt on Raspberry Pi OS, not Poetry. These are pre-installed or available via:
```bash
sudo apt install python3-picamera2 python3-rpi.gpio python3-smbus
pip install luma.oled
```

### 7. Run Pi Client

The Pi client uses command-line arguments for configuration:

```bash
# View all options
poetry run python -m pi.main --help

# Connect to server with mock hardware (testing)
poetry run python -m pi.main --host 192.168.1.100

# Connect to server with real hardware (production)
poetry run python -m pi.main --host 192.168.1.100 --real-hardware

# With verbose logging
poetry run python -m pi.main --host 192.168.1.100 --real-hardware -v
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | localhost | Server brain hostname/IP |
| `--port` | 6765 | Server WebSocket port |
| `--real-hardware` | false | Use real Pi hardware (not mocks) |
| `-v, --verbose` | false | Enable debug logging |

---

## Emulator Deployment

The emulator simulates the Pi hardware for development/testing.

### 1. Install Dependencies

```bash
# Same as server setup
poetry install
```

### 2. Run Emulator

```bash
poetry run python -m emulator
```

Emulator UI: `http://localhost:6080`

### 3. Webcam/Microphone Permissions

**Linux:**
```bash
# Add user to video/audio groups
sudo usermod -a -G video,audio $USER
# Log out and back in
```

**macOS:**
Grant camera/microphone access when prompted.

**Windows (WSL2):**
Webcam passthrough requires additional setup. Use mock video instead.

---

## Systemd Services

### Server Service

Create `/etc/systemd/system/murph-server.service`:

```ini
[Unit]
Description=Murph Robot Server
After=network.target

[Service]
Type=simple
User=murph
WorkingDirectory=/home/murph/murph
Environment=PATH=/home/murph/.local/bin:/usr/bin
ExecStart=/home/murph/.local/bin/poetry run python -m server.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable murph-server
sudo systemctl start murph-server
```

### Pi Client Service

Create `/etc/systemd/system/murph-pi.service`:

```ini
[Unit]
Description=Murph Robot Pi Client
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/murph
Environment=PATH=/home/pi/.local/bin:/usr/bin
# Configure server IP below (replace 192.168.1.100 with your server)
ExecStart=/home/pi/.local/bin/poetry run python -m pi.main --host 192.168.1.100 --real-hardware
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable murph-pi
sudo systemctl start murph-pi
```

### View Logs

```bash
# Server logs
sudo journalctl -u murph-server -f

# Pi client logs
sudo journalctl -u murph-pi -f
```

---

## Network Configuration

### Firewall (Server)

Open required ports:

```bash
# UFW (Ubuntu)
sudo ufw allow 6765/tcp  # WebSocket
sudo ufw allow 6081/tcp  # Dashboard

# firewalld (Fedora/RHEL)
sudo firewall-cmd --permanent --add-port=6765/tcp
sudo firewall-cmd --permanent --add-port=6081/tcp
sudo firewall-cmd --reload
```

### Static IP (Raspberry Pi)

Edit `/etc/dhcpcd.conf`:
```ini
interface wlan0
static ip_address=192.168.1.50/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

### mDNS/Avahi (Optional)

For `murph.local` hostname resolution:

```bash
# Server
sudo apt install avahi-daemon
sudo systemctl enable avahi-daemon

# Pi can then connect to:
MURPH_SERVER_HOST=server-hostname.local
```

---

## Troubleshooting

### Server Won't Start

```bash
# Check logs
poetry run python -m server.main 2>&1 | head -50

# Common issues:
# - Port 6765 in use: kill existing process or change port
# - Missing dependencies: poetry install
# - Ollama not running: ollama serve
```

### Pi Can't Connect to Server

```bash
# Test network connectivity
ping 192.168.1.100

# Test WebSocket port
nc -zv 192.168.1.100 6765

# Check firewall on server
sudo ufw status

# Verify .env settings
cat .env | grep MURPH_SERVER
```

### I2C Devices Not Detected

```bash
# Check I2C is enabled
ls /dev/i2c*

# Scan for devices
sudo i2cdetect -y 1

# If empty, check wiring and device power
```

### Camera Not Working

```bash
# Test camera
libcamera-hello

# Check camera connection
vcgencmd get_camera

# Enable in config
sudo raspi-config  # Interface Options → Camera
```

### Audio Issues

```bash
# List audio devices
arecord -l  # Input devices
aplay -l    # Output devices

# Test recording
arecord -d 5 -f S16_LE -r 16000 test.wav
aplay test.wav

# Check I2S overlay in /boot/firmware/config.txt
```

### Service Not Starting

```bash
# Check service status
sudo systemctl status murph-server

# View detailed logs
sudo journalctl -u murph-server -n 100 --no-pager

# Test manual start
sudo -u murph /home/murph/.local/bin/poetry run python -m server.main
```

### Memory Issues on Pi

```bash
# Check memory usage
free -h

# Reduce video resolution in constants.py:
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_FPS = 5
```
