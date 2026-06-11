# WiSentry — Windows 10/11 Setup Guide

Follow this top to bottom. Every command is copy-pasteable. You never need
to edit code except the clearly marked CONFIGURATION blocks in the firmware.

---

## Part A — Python environment (laptop side)

1. **Install Python 3.10 or newer.**
   Download from https://www.python.org/downloads/windows/ and run the
   installer. On the first screen **tick the checkbox "Add python.exe to
   PATH"** — this matters — then click *Install Now*.

2. **Verify.** Open *Command Prompt* (press `Win`, type `cmd`, Enter):
   ```bat
   python --version
   ```
   You should see `Python 3.1x.x`. If you see an error or the Microsoft
   Store opens, re-run the installer and tick the PATH checkbox.

3. **Get the project.**
   ```bat
   cd %USERPROFILE%\Documents
   git clone https://github.com/Arjunsk1291/wisentry.git
   cd wisentry
   ```
   (No git? Download the ZIP from the GitHub page, extract it, and `cd`
   into the folder instead.)

4. **Install dependencies.**
   ```bat
   pip install -r requirements.txt
   ```
   This takes a few minutes (PyTorch is large). Wait for it to finish.

5. **Check everything.**
   ```bat
   python setup_check.py
   ```
   Every line must say `[ OK ]`. If anything says `[FAIL]`, the message
   tells you the exact fix; also see `docs/troubleshooting.md`.

6. **First run, no hardware needed.**
   ```bat
   python main.py --simulate
   ```
   Your browser should be opened by you at **http://localhost:8050** —
   you'll see the dashboard with a red "SIMULATION MODE" banner and a
   simulated person walking in after 5 seconds. `Ctrl+C` in the command
   window stops it.

   > **Windows Firewall popup:** the first run may ask whether Python can
   > use the network. Tick **Private networks** and click *Allow access*.
   > If you skipped it, see troubleshooting entry #12.

---

## Part B — Flashing the ESP32 boards

You will flash **one transmitter** and **one or more receivers**. The
procedure is identical except for which sketch you open and one number
you change per receiver.

### B1. Install the USB driver

Most ESP32 DevKit boards use one of two USB chips. Look at the small chip
near the USB connector:

| Chip printed on it | Driver |
|---|---|
| `CP2102` / `CP210x` | https://www.silabs.com/developer-tools/usb-to-uart-bridge-vcp-drivers — download "CP210x Windows Drivers", unzip, run `CP210xVCPInstaller_x64.exe` |
| `CH340` / `CH9102` | https://www.wch-ic.com/downloads/CH341SER_EXE.html — download and run `CH341SER.EXE`, click *Install* |

Plug the ESP32 in with a **data-capable** USB cable (some charging cables
have no data wires — if no new device appears, try another cable first).
Open *Device Manager* (`Win+X` → Device Manager) → **Ports (COM & LPT)**:
you should see something like `Silicon Labs CP210x (COM5)`. **Note the
COM number.**

### B2. Install Arduino IDE and the ESP32 board package

1. Download **Arduino IDE 2.x** from https://www.arduino.cc/en/software
   (Windows MSI installer), install with defaults, open it.
2. *File → Preferences* → in **Additional boards manager URLs** paste:
   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
   Click OK.
3. *Tools → Board → Boards Manager…* → search **esp32** → install
   **"esp32 by Espressif Systems"** (takes several minutes).

### B3. Flash the TRANSMITTER

1. *File → Open…* → select
   `wisentry\firmware\csi_transmitter\csi_transmitter.ino`.
2. *Tools → Board → esp32 →* **ESP32 Dev Module**. Leave every other
   Tools menu item at its default.
3. *Tools → Port →* your COM port from step B1.
4. (Optional) edit the CONFIGURATION block: WiFi name, password, channel.
   The defaults work as-is.
5. Click the **→ Upload** button. You'll see compiling (1–3 min), then
   `Connecting....`, then `Writing at 0x...`, then `Hard resetting`.

   > **Stuck at `Connecting....._____....`?** Hold the **BOOT** button on
   > the board while it retries, release when writing starts.

6. *Tools → Serial Monitor*, set **115200 baud** (dropdown bottom-right).
   You should see:
   ```text
   Access point 'WiSentry' up on channel 6, AP IP = 192.168.4.1
   Transmitting sounding packets every 10 ms.
   alive: 100 packets sent, stations joined: 0
   ```
   Label this board **TX** with a sticker. Done — it now just needs USB
   power (any phone charger works).

### B4. Flash each RECEIVER

1. Plug in the next board (you can unplug TX — power it from a charger).
2. *File → Open…* →
   `wisentry\firmware\csi_receiver\csi_receiver.ino`.
3. In the CONFIGURATION block set the receiver's identity:
   ```cpp
   static const uint8_t DEVICE_ID = 1;   // 2 for your second receiver, etc.
   ```
4. Board = **ESP32 Dev Module**, Port = its COM port, click **Upload**.
5. Open the Serial Monitor (115200). With the TX powered you should see:
   ```text
   WiSentry receiver RX-1 starting...
   Joining 'WiSentry'....
   Connected. My IP = 192.168.4.2, RSSI = -45 dBm
   CSI capture enabled. Streaming to laptop.
   csi captured: 87, sent: 87, dropped (ring full): 0, wifi rssi: -45
   ```
   The `csi captured` number must keep climbing — that is live CSI.
6. Label the board **RX-1** (RX-2, …) and repeat for each receiver.

### B5. Connect the laptop and go live

1. Connect your laptop's WiFi to the **`WiSentry`** network
   (password `wisentry-csi` unless you changed it).
2. In the project folder:
   ```bat
   python main.py
   ```
3. Open **http://localhost:8050**. No red banner this time — the Device
   Table should list every receiver as ONLINE with a climbing packet
   count. Walk around the room and watch the waveform respond.
4. Expect rough detection at first: the models shipped in this repo were
   trained on synthetic data. Collect real data and retrain (see
   `docs/user_manual.md`, chapter 9) to reach the accuracy targets.

---

## Where to place the boards

See `docs/placement_guide.md` for room diagrams. Short version: TX on one
side of the room, receivers on the opposite side at chest height, person
walks between them.
