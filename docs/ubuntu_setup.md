# WiSentry — Ubuntu Setup Guide

Differences from the Windows guide only; the project workflow is
identical. Tested against Ubuntu 22.04/24.04.

## Part A — Python environment

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
cd ~
git clone https://github.com/Arjunsk1291/wisentry.git
cd wisentry
python3 -m venv .venv
source .venv/bin/activate          # re-run this in every new terminal
pip install -r requirements.txt
python setup_check.py              # every line must say [ OK ]
python main.py --simulate          # then open http://localhost:8050
```

Notes:
- Inside the venv, `python` and `pip` refer to the venv copies — the
  docs' `python` commands work as written.
- No firewall prompt appears on stock Ubuntu (ufw is off by default).
  If you enabled ufw: `sudo ufw allow 5566/udp`.

## Part B — Flashing the ESP32 boards

### B1. Serial port permission (the #1 Ubuntu gotcha)

Your user must be in the `dialout` group to open `/dev/ttyUSB0`:

```bash
sudo usermod -aG dialout $USER
```

**Log out and back in** (or reboot) for this to take effect. Without it,
every upload fails with `Permission denied: '/dev/ttyUSB0'`.

Also: if the IDE can't see the board and `lsusb` shows the chip, recent
Ubuntu ships `brltty` which hijacks CH340 chips — remove it:

```bash
sudo apt remove -y brltty
```

then re-plug the board.

### B2. Drivers

None needed — CP210x and CH340 drivers are in the Ubuntu kernel. Plug
the board in and confirm:

```bash
ls /dev/ttyUSB*      # expect /dev/ttyUSB0
```

### B3. Arduino IDE

1. Download the Linux AppImage from https://www.arduino.cc/en/software
2. ```bash
   chmod +x arduino-ide_*.AppImage
   ./arduino-ide_*.AppImage
   ```
   (If it fails to start on 22.04: `sudo apt install -y libfuse2`.)
3. From here the steps are identical to `windows_setup.md` Part B2–B4 —
   add the ESP32 boards URL, install "esp32 by Espressif Systems",
   open each sketch, board = **ESP32 Dev Module**, port =
   **/dev/ttyUSB0**, Upload, check the Serial Monitor at 115200.

### Command-line alternative (no IDE)

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
export PATH="$PATH:$HOME/bin"
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
    https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
# compile + flash the transmitter:
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/csi_transmitter
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/csi_transmitter
# serial monitor:
arduino-cli monitor -p /dev/ttyUSB0 -c baudrate=115200
```

## Part C — Going live

Same as Windows Part B5: join the `WiSentry` WiFi network
(NetworkManager: click the WiFi icon → WiSentry → password
`wisentry-csi`), then:

```bash
source .venv/bin/activate
python main.py
```

Dashboard at http://localhost:8050.
