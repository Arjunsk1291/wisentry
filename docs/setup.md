# WiSentry — Setup Guide

One guide for getting WiSentry running: Windows, Ubuntu, and fixes for
common problems.

- [1. Windows 10/11](#1-windows-1011)
- [2. Ubuntu](#2-ubuntu)
- [3. Troubleshooting](#3-troubleshooting)

## 1. Windows 10/11

Follow this top to bottom. Every command is copy-pasteable. You never need
to edit code except the clearly marked CONFIGURATION blocks in the firmware.

---

### Part A — Python environment (laptop side)

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
   tells you the exact fix; also see the Troubleshooting section.

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

### Part B — Flashing the ESP32 boards

You will flash **one transmitter** and **one or more receivers**. The
procedure is identical except for which sketch you open and one number
you change per receiver.

#### B1. Install the USB driver

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

#### B2. Install Arduino IDE and the ESP32 board package

1. Download **Arduino IDE 2.x** from https://www.arduino.cc/en/software
   (Windows MSI installer), install with defaults, open it.
2. *File → Preferences* → in **Additional boards manager URLs** paste:
   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
   Click OK.
3. *Tools → Board → Boards Manager…* → search **esp32** → install
   **"esp32 by Espressif Systems"** (takes several minutes).

#### B3. Flash the TRANSMITTER

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

#### B4. Flash each RECEIVER

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

#### B5. Connect the laptop and go live

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

### Where to place the boards

See `docs/placement_guide.md` for room diagrams. Short version: TX on one
side of the room, receivers on the opposite side at chest height, person
walks between them.

## 2. Ubuntu

Differences from the Windows guide only; the project workflow is
identical. Tested against Ubuntu 22.04/24.04.

### Part A — Python environment

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

### Part B — Flashing the ESP32 boards

#### B1. Serial port permission (the #1 Ubuntu gotcha)

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

#### B2. Drivers

None needed — CP210x and CH340 drivers are in the Ubuntu kernel. Plug
the board in and confirm:

```bash
ls /dev/ttyUSB*      # expect /dev/ttyUSB0
```

#### B3. Arduino IDE

1. Download the Linux AppImage from https://www.arduino.cc/en/software
2. ```bash
   chmod +x arduino-ide_*.AppImage
   ./arduino-ide_*.AppImage
   ```
   (If it fails to start on 22.04: `sudo apt install -y libfuse2`.)
3. From here the steps are identical to the Windows section, Part B2–B4 —
   add the ESP32 boards URL, install "esp32 by Espressif Systems",
   open each sketch, board = **ESP32 Dev Module**, port =
   **/dev/ttyUSB0**, Upload, check the Serial Monitor at 115200.

#### Command-line alternative (no IDE)

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

### Part C — Going live

Same as Windows Part B5: join the `WiSentry` WiFi network
(NetworkManager: click the WiFi icon → WiSentry → password
`wisentry-csi`), then:

```bash
source .venv/bin/activate
python main.py
```

Dashboard at http://localhost:8050.

## 3. Troubleshooting

30 real problems, grouped by stage. Find your symptom, apply the fix.

### A. Python / installation

**1. `python` is not recognized (Windows).**
Python isn't on PATH. Re-run the installer, tick *Add python.exe to
PATH*, reopen Command Prompt. Quick test: `py --version` often works
even when `python` doesn't — if so, use `py` everywhere.

**2. Typing `python` opens the Microsoft Store.**
Settings → Apps → *App execution aliases* → turn **off** both
`python.exe` and `python3.exe` aliases, then reinstall Python with PATH.

**3. `pip install -r requirements.txt` fails on torch.**
Usually disk space (torch needs ~3 GB free) or a 32-bit Python. Install
64-bit Python, free space, retry. Corporate proxy? Add
`--proxy http://user:pass@proxy:port`.

**4. `pip` works but `setup_check.py` says a package is MISSING.**
You have two Pythons; pip installed into the other one. Run
`python -m pip install -r requirements.txt` (note the `python -m`).

**5. `setup_check.py`: "UDP port 5566 is busy".**
A previous WiSentry run is still alive. Windows:
`taskkill /f /im python.exe` (closes ALL Python). Ubuntu:
`pkill -f main.py`. Or change `udp_listen_port` in config.yaml (then
also change `LAPTOP_UDP_PORT` in the receiver firmware).

**6. `setup_check.py`: "TCP port 8050 is busy".**
Something else serves on 8050. Change `dashboard.port` in config.yaml
to e.g. 8060 and browse to http://localhost:8060.

**7. `ConfigError: config.yaml is not valid YAML`.**
You edited config.yaml and broke indentation. YAML needs spaces, never
tabs, and the exact `key: value` spacing. Restore from git:
`git checkout -- config.yaml` and redo the edit carefully.

### B. Simulation mode

**8. `python main.py --simulate` exits immediately with an import error.**
You're not in the project folder. `cd` into `wisentry` first; the prompt
should show the folder name.

**9. Dashboard page is blank / spinning.**
Give it ~5 s after start. Then hard-refresh (`Ctrl+F5`). Still blank:
check the terminal for a traceback and report it.

**10. Simulation runs but presence never triggers.**
The detector learns its empty-room baseline from the first ~2 s of
frames. If you changed `simulation.frame_rate_hz` to something very low
(<20), windows take too long to fill. Restore defaults in config.yaml.

**11. `events_*.csv` files pile up in logs/.**
One per run, by design. Delete freely; set `logging.log_events: false`
to stop creating them.

### C. Network / laptop side (live mode)

**12. Receivers say "sent: N" climbing, but the Device Table is empty.**
90% of the time: Windows Firewall blocked Python. Settings → *Windows
Defender Firewall* → *Allow an app through firewall* → tick **Private**
for Python. Or re-trigger the prompt: delete the rule and rerun
`python main.py`. Also confirm the laptop is on the **WiSentry** WiFi.

**13. Device Table shows the receiver OFFLINE with a frozen packet count.**
The receiver stopped sending or left the network. Check its serial
monitor: if it prints `WiFi lost — reconnecting...` repeatedly, move it
closer to the TX (check the RSSI in its serial output; below −80 dBm is
too far).

**14. `udp_server: cannot bind 0.0.0.0:5566`.**
Same as #5 — another instance is running.

**15. Frames arrive but "parse errors" climbs in the shutdown summary.**
Something else is spraying UDP at port 5566, or firmware/laptop protocol
versions diverged. Reflash receivers from the current repo so firmware
and `backend/csi_parser.py` match (both must implement protocol v1).

**16. Laptop on WiSentry WiFi loses internet.**
Expected — the WiSentry SoftAP has no internet uplink. Sensing doesn't
need it. Use ethernet for internet, or run the system on your home WiFi
instead (see the station-mode note in `docs/user_manual.md` ch. 6).

### D. Flashing / ESP32 hardware

**17. No COM port appears when plugging in the board.**
In order: try another USB cable (most failures are charge-only cables);
try another USB port; install the right driver (CP210x vs CH340 — read
the chip, see setup guide B1). Device Manager should show *Ports (COM &
LPT)*. Ubuntu: see the `brltty` fix in the Ubuntu section below.

**18. Upload stuck at `Connecting........_____`.**
Hold the **BOOT** button on the board during `Connecting...`, release
once `Writing at 0x...` appears. Some clone boards need this every time.

**19. `A fatal error occurred: MD5 of file does not match data in flash`.**
Bad cable or overlong USB hub chain. Plug directly into the laptop with
a short cable and reflash.

**20. Board flashes fine but Serial Monitor shows garbage characters.**
Wrong baud rate. Set the Serial Monitor dropdown to **115200**.

**21. Serial Monitor shows a reboot loop (`rst:0x... boot:0x...` forever).**
Power problem (brownout) or corrupted flash. Use a USB port that can
supply 500 mA (not a passive hub), then reflash with *Tools → Erase All
Flash Before Sketch Upload → Enabled* once.

**22. `esp_wifi.h: No such file or directory` when compiling.**
The selected board isn't an ESP32 (you're compiling for Arduino Uno or
similar). *Tools → Board → esp32 → ESP32 Dev Module.*

**23. Receiver prints `Joining 'WiSentry'.....` dots forever.**
The transmitter isn't powered/running, or SSID/password were edited on
one board but not the other, or the receiver is out of range. Power the
TX first, check its serial output says the AP is up, bring the receiver
into the same room.

**24. Receiver joins but `csi captured` stays at 0.**
The TX must actually be transmitting (its serial prints `alive: N
packets sent` with N climbing). If TX is fine and CSI is still 0,
reflash the receiver — `CSI capture enabled.` must appear in its boot
log; if instead you see an `ESP_ERROR_CHECK failed` line, the flashed
core version is too old: update the esp32 boards package (≥ 2.0).

**25. `dropped (ring full)` climbing fast on a receiver.**
The receiver hears more frames than it can forward (busy channel).
Harmless at small counts. If it's huge, choose a quieter WiFi channel on
the TX (`ACCESS_POINT_CHANNEL`, use 1/6/11) and reflash both boards.

### E. Detection quality

**26. Presence triggers when the room is empty.**
(a) The baseline was learned while someone was in the room — restart
`python main.py` with the room empty for the first 10 s. (b) A fan,
curtain, or pet is moving — see placement guide. (c) Raise
`detection.rule_motion_threshold` slightly (e.g. 1.2 → 1.8) if you're
on rule-based mode.

**27. Presence misses a person sitting very still.**
Breathing detection needs ~5 s of accumulated signal — give it time.
If still missed: the person may be outside the TX↔RX paths (move
devices per the placement guide), or lower
`detection.rule_motion_threshold`.

**28. Pose label flickers or is mostly wrong on real hardware.**
Expected with the shipped models — they are trained on **synthetic**
data and validate the pipeline, not your room. Collect real labeled
data (`python main.py --collect --label standing`, etc.) and retrain
(`python models/train_all.py --real` once Phase 6 lands; see
docs/dev/PROJECT_LOG.md for status). Until then treat pose as a demo.

**29. Pose stuck on "walking" whenever anyone moves at all.**
Lower `detection.pose_confidence_threshold` (0.6 → 0.5) and confirm at
least 2 receivers are ONLINE — single-receiver pose is unreliable by
design (see Coverage Advisor).

**30. Everything works, then degrades hours later.**
Someone moved a board, or the WiFi channel got congested (neighbors).
Restart `python main.py` to re-baseline; consider a different channel.
If the laptop's CPU is pegged, check that only one `main.py` is running.

---

Still stuck? Read the terminal output carefully — every WiSentry error
message says what to fix — then check docs/dev/PROJECT_LOG.md for known issues,
then open a GitHub issue with: your OS, the exact command, and the full
terminal output.
