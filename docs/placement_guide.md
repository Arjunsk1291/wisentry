# WiSentry — Device Placement Guide

WiFi CSI sensing works by measuring how a body disturbs the radio paths
between the transmitter (TX) and each receiver (RX). Placement rule of
thumb: **the person should move through the space between TX and RX**,
not behind either of them.

## General rules

1. Mount boards at **chest height (1.0–1.4 m)** — tables, shelves, or
   tape them to a wall. Floor placement halves sensitivity.
2. Keep TX and each RX **3–6 m apart**. Closer than 2 m saturates;
   farther than ~8 m weakens the signal (watch RSSI in the Device Table:
   −40 to −65 dBm is the sweet spot).
3. Keep boards **away from large metal objects** (fridges, radiators,
   metal cabinets) — at least 50 cm.
4. Antenna side (the etched zig-zag end of the board) pointing **into
   the room**, not into the wall.
5. After any re-placement, **restart `python main.py`** so the empty-room
   baseline is re-learned — and stay out of the room for the first
   ~10 seconds.

## 1 receiver — presence only (~15 m²)

```text
+------------------------------------------+
|                                          |
| TX >                              < RX-1 |
|        . . . sensing zone . . .          |
|        .  (between the boards) .         |
|        . . . . . . . . . . . .           |
|                                          |
+------------------------------------------+
```

One radio path. Anyone crossing or lingering near the TX↔RX-1 line is
detected. Corners far from the line are weak spots.

## 2 receivers — presence + pose (~25 m²)

```text
+------------------------------------------+
| RX-1 <                                   |
|         \                                |
|          \   two crossing                |
| TX >       paths cover the               |
|          /   middle well                 |
|         /                                |
| RX-2 <                                   |
+------------------------------------------+
   TX on one wall, receivers on the opposite wall's corners.
```

Two differently-angled paths give the pose model independent views —
this is the recommended minimum for reliable pose classification.

## 3 receivers — full system (~35 m²)

```text
+------------------------------------------+
| RX-1 <                            > RX-3 |
|                                          |
|              (room center =              |
| TX >          strongest zone)            |
|                                          |
| RX-2 <                                   |
+------------------------------------------+
```

Three paths triangulate the room. This enables the experimental
skeleton tier and makes the room-map position estimate meaningful.

## Coverage summary

| Receivers online | Capability | Coverage guide |
|---|---|---|
| 1 | presence only | ~15 m² |
| 2 | + pose | ~25 m² |
| 3+ | + skeleton (experimental) | ~35 m²+, scales with layout |

The dashboard's **Coverage Advisor** panel shows this live, based on how
many receivers are actually online.

## What degrades sensing

- **Through-wall setups**: works at reduced sensitivity through drywall;
  brick/concrete mostly blocks it. Keep the system in one room at first.
- **Other 2.4 GHz traffic on the same channel**: pick channel 1, 6, or 11
  — whichever your home WiFi does *not* use (change
  `ACCESS_POINT_CHANNEL` in the transmitter firmware).
- **Pets**: a large dog reads like a small person. Expect false
  positives until you retrain with your own collected data (Phase 6).
- **Fans / swaying curtains** near a board: constant micro-motion raises
  the noise floor. Move the board or the object.
