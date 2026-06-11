/*
 * WiSentry CSI TRANSMITTER (device_id 0)
 * --------------------------------------
 * Role: floods the air with WiFi packets at a steady rate so the
 * receiver ESP32s can measure CSI from them. This board does NOT
 * capture CSI itself — it is the "radio illuminator" of the room.
 *
 * Default topology (recommended for first setup):
 *   This board creates a WiFi access point (SSID below). The receiver
 *   ESP32s AND your laptop both join that network. No router needed.
 *
 * Board:    ESP32-WROOM-32 / ESP32 DevKitC ("ESP32 Dev Module" in IDE)
 * Core:     arduino-esp32 (board package "esp32 by Espressif Systems")
 * Wiring:   USB cable only (power + flashing).
 *
 * Flash this exact file; the only lines you may edit are in the
 * CONFIGURATION block.
 */

#include <WiFi.h>          // WiFi.softAP and network classes
#include <WiFiUdp.h>       // simple UDP transmit

/* ------------------------- CONFIGURATION ------------------------- */
/* Network name and password this transmitter creates. The receivers'
 * firmware and your laptop must use the same values.                 */
static const char *ACCESS_POINT_SSID = "WiSentry";
static const char *ACCESS_POINT_PASSWORD = "wisentry-csi";  /* >= 8 chars */

/* WiFi channel of the access point. All CSI sensing happens on this
 * channel. 1, 6, or 11 are the usual least-congested choices.        */
static const int ACCESS_POINT_CHANNEL = 6;

/* How often to transmit a sounding packet. 10 ms = 100 packets/s.
 * The receivers' achievable CSI rate is bounded by this.             */
static const unsigned long TRANSMIT_INTERVAL_MS = 10;

/* UDP port the sounding packets are broadcast to. The payload content
 * is irrelevant — receivers measure the radio channel, not the data.
 * (Port 5567 != 5566: the laptop's CSI port stays clean.)            */
static const uint16_t SOUNDING_UDP_PORT = 5567;

/* Onboard LED (GPIO 2 on most DevKit boards): blinks while running.  */
static const int STATUS_LED_PIN = 2;
static const unsigned long LED_BLINK_INTERVAL_MS = 1000;
/* ------------------------------------------------------------------ */

/* The broadcast address of the SoftAP subnet (192.168.4.0/24 by
 * default on ESP32), so every joined station hears the packet.       */
static const IPAddress BROADCAST_ADDRESS(192, 168, 4, 255);

static WiFiUDP soundingUdpSocket;
static unsigned long lastTransmitMillis = 0;
static unsigned long lastBlinkMillis = 0;
static bool ledState = false;
static uint32_t transmittedPacketCount = 0;

/* Tiny fixed payload; receivers never read it.                       */
static const uint8_t SOUNDING_PAYLOAD[8] =
    {'W', 'S', 'N', 'T', 'R', 'Y', 0, 0};

void setup() {
  /* Serial monitor at 115200 baud for status messages.               */
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("WiSentry transmitter starting...");

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  /* Create the access point on a fixed channel. Receivers and the
   * laptop join this network.                                        */
  bool accessPointStarted = WiFi.softAP(
      ACCESS_POINT_SSID, ACCESS_POINT_PASSWORD, ACCESS_POINT_CHANNEL);
  if (!accessPointStarted) {
    Serial.println("FATAL: could not start the access point. "
                   "Power-cycle the board and reflash.");
    while (true) { delay(1000); }  /* halt with a readable error      */
  }
  Serial.print("Access point '");
  Serial.print(ACCESS_POINT_SSID);
  Serial.print("' up on channel ");
  Serial.print(ACCESS_POINT_CHANNEL);
  Serial.print(", AP IP = ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Transmitting sounding packets every 10 ms.");
}

void loop() {
  unsigned long nowMillis = millis();

  /* Steady-rate transmission without blocking delay():               */
  if (nowMillis - lastTransmitMillis >= TRANSMIT_INTERVAL_MS) {
    lastTransmitMillis = nowMillis;
    soundingUdpSocket.beginPacket(BROADCAST_ADDRESS, SOUNDING_UDP_PORT);
    soundingUdpSocket.write(SOUNDING_PAYLOAD, sizeof(SOUNDING_PAYLOAD));
    soundingUdpSocket.endPacket();
    transmittedPacketCount++;
  }

  /* Heartbeat LED + once-per-blink status line.                      */
  if (nowMillis - lastBlinkMillis >= LED_BLINK_INTERVAL_MS) {
    lastBlinkMillis = nowMillis;
    ledState = !ledState;
    digitalWrite(STATUS_LED_PIN, ledState ? HIGH : LOW);
    Serial.print("alive: ");
    Serial.print(transmittedPacketCount);
    Serial.print(" packets sent, stations joined: ");
    Serial.println(WiFi.softAPgetStationNum());
  }
}
