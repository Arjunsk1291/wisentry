# WiSentry — Troubleshooting

30 real problems, grouped by stage. Find your symptom, apply the fix.

## A. Python / installation

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

## B. Simulation mode

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

## C. Network / laptop side (live mode)

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

## D. Flashing / ESP32 hardware

**17. No COM port appears when plugging in the board.**
In order: try another USB cable (most failures are charge-only cables);
try another USB port; install the right driver (CP210x vs CH340 — read
the chip, see setup guide B1). Device Manager should show *Ports (COM &
LPT)*. Ubuntu: see the `brltty` fix in `docs/ubuntu_setup.md`.

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

## E. Detection quality

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
PROJECT_LOG.md for status). Until then treat pose as a demo.

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
message says what to fix — then check PROJECT_LOG.md for known issues,
then open a GitHub issue with: your OS, the exact command, and the full
terminal output.
