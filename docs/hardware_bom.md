# WiSentry — Hardware Bill of Materials

Everything is off-the-shelf. No soldering, no extra sensors, no SBCs.

## Minimum build (presence detection, ~15 m²)

| # | Item | Qty | Typical price (2026) | Notes |
|---|------|-----|------------------|-------|
| 1 | ESP32 DevKitC / ESP32-WROOM-32 dev board | 2 | $4–8 each | 1 transmitter + 1 receiver. Must be the **classic ESP32** (see "Which board" below). |
| 2 | USB-A → Micro-USB **data** cable | 2 | $2–4 each | Many bundled cables are charge-only — buy ones marked "data" or "sync". |
| 3 | USB charger / power bank, 5 V ≥ 1 A | 2 | $5–10 each | Any phone charger. One per board after flashing. |

**Minimum total: roughly $25–40.**

## Recommended build (pose detection, ~25 m²)

Add one more of items 1–3 (a third ESP32 as RX-2): **+$10–20**.

## Full build (skeleton experiments, ~35 m²)

Four ESP32s total (TX + 3 RX): **roughly $45–75 all-in**.

You also need a Windows 10/11 or Ubuntu laptop with WiFi — a mid-range
i5 from the last ~6 years is plenty (the whole pipeline targets <40% CPU).

## Which board exactly ("classic ESP32" requirement)

The CSI API used by the firmware targets the original ESP32 silicon.
**Buy boards described as one of:**

- "ESP32 DevKitC"
- "ESP32-WROOM-32" / "ESP32-WROOM-32D" / "ESP32-WROOM-32U"
- "ESP32 Dev Module, 38-pin" (the common AliExpress/Amazon clone)

**Avoid for this project:** ESP32-**S2** (no usable CSI), ESP32-**C3**,
ESP32-**S3**, ESP32-**C6**, ESP8266, and Arduino boards without WiFi.
The letter suffix matters more than the brand.

Clone boards are fine — that is what the firmware was written for. The
only practical difference is the USB chip (CP2102 vs CH340), which just
changes which driver you install (see the setup guides).

## Where to buy

| Source | Notes |
|--------|-------|
| AliExpress ("ESP32 DevKitC WROOM-32") | Cheapest ($3–5), 2–4 week shipping |
| Amazon ("ESP32 development board 38 pin") | $7–10, fast shipping, easy returns |
| Espressif official store / Mouser / DigiKey | Genuine DevKitC ~$10, guaranteed authentic |

Buy at least one spare — at these prices a dead-on-arrival board should
never block your week.

## Explicitly NOT needed

No cameras, no PIR sensors, no mmWave radars, no Raspberry Pi, no
antennas, no soldering iron, no breadboards, no jumper wires, no cloud
subscription. The ESP32 boards and USB power are the entire system.
