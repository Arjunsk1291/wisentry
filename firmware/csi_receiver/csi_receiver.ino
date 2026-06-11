/*
 * WiSentry CSI RECEIVER (device_id 1..N)
 * --------------------------------------
 * Role: joins the WiSentry transmitter's WiFi network, captures CSI
 * (Channel State Information) from every packet it hears, and streams
 * each measurement to the laptop as one UDP datagram in WiSentry wire
 * protocol v1 (see ENGINEERING_SPEC.md §5.2 — must match backend/csi_parser.py).
 *
 * Wire protocol v1, little-endian, 12-byte header + 128 bytes CSI:
 *   uint8  protocol_version = 1
 *   uint8  device_id
 *   uint32 sequence_number
 *   uint32 esp32_timestamp_us
 *   int8   rssi_dbm
 *   uint8  subcarrier_count = 64
 *   int8[128] interleaved (imaginary, real) pairs
 *
 * Board:    ESP32-WROOM-32 / ESP32 DevKitC ("ESP32 Dev Module" in IDE)
 * Core:     arduino-esp32 (board package "esp32 by Espressif Systems")
 * Wiring:   USB cable only (power + flashing).
 *
 * Flash one copy per receiver, changing only DEVICE_ID (1, 2, 3...).
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"      /* esp_wifi_set_csi* low-level CSI API     */

/* ------------------------- CONFIGURATION ------------------------- */
/* Unique identity of THIS receiver: 1 for RX-1, 2 for RX-2, ...      */
static const uint8_t DEVICE_ID = 1;

/* Network created by the WiSentry transmitter (or your home WiFi if
 * you run the transmitter in station mode — see docs).               */
static const char *NETWORK_SSID = "WiSentry";
static const char *NETWORK_PASSWORD = "wisentry-csi";

/* Where to send CSI datagrams. 192.168.4.255 broadcasts to everyone
 * on the transmitter's SoftAP subnet, so you do NOT need to know the
 * laptop's IP. If you use your home WiFi instead, set this to the
 * laptop's address printed by `ipconfig` / `ip addr`.                */
static const IPAddress LAPTOP_ADDRESS(192, 168, 4, 255);
static const uint16_t LAPTOP_UDP_PORT = 5566;

static const int STATUS_LED_PIN = 2;
static const unsigned long STATUS_REPORT_INTERVAL_MS = 1000;
/* ------------------------------------------------------------------ */

/* Protocol v1 constants — keep in lockstep with backend/csi_parser.py */
static const uint8_t PROTOCOL_VERSION = 1;
static const uint8_t SUBCARRIER_COUNT = 64;
static const int CSI_PAYLOAD_BYTES = SUBCARRIER_COUNT * 2;  /* 128    */

/* One serialized datagram. ESP32 is little-endian, so a packed struct
 * lays out exactly the bytes the wire protocol requires.             */
typedef struct __attribute__((packed)) {
  uint8_t protocolVersion;
  uint8_t deviceId;
  uint32_t sequenceNumber;
  uint32_t timestampMicros;
  int8_t rssiDbm;
  uint8_t subcarrierCount;
  int8_t csiPairs[CSI_PAYLOAD_BYTES];
} CsiDatagram;

/* The CSI callback runs inside the WiFi driver task — it must be fast
 * and must not call the network stack. It writes into this small ring
 * buffer; loop() drains the ring and does the UDP sends.             */
static const int RING_SLOTS = 8;
static volatile CsiDatagram ringBuffer[RING_SLOTS];
static volatile int ringWriteIndex = 0;
static volatile int ringReadIndex = 0;
static volatile uint32_t capturedFrameCount = 0;
static volatile uint32_t droppedFrameCount = 0;

static WiFiUDP csiUdpSocket;
static uint32_t sentDatagramCount = 0;
static unsigned long lastStatusMillis = 0;

/* Called by the WiFi driver for every received frame with CSI.       */
static void onCsiCaptured(void *context, wifi_csi_info_t *csiInfo) {
  (void)context;
  if (csiInfo == NULL || csiInfo->buf == NULL) {
    return;  /* defensive: driver handed us nothing                   */
  }
  if (csiInfo->len < CSI_PAYLOAD_BYTES) {
    return;  /* frame too short for 64 subcarriers (non-LLTF frame)   */
  }
  int nextWriteIndex = (ringWriteIndex + 1) % RING_SLOTS;
  if (nextWriteIndex == ringReadIndex) {
    droppedFrameCount++;  /* ring full: loop() is behind — drop       */
    return;
  }
  volatile CsiDatagram *slot = &ringBuffer[ringWriteIndex];
  slot->protocolVersion = PROTOCOL_VERSION;
  slot->deviceId = DEVICE_ID;
  slot->sequenceNumber = ++capturedFrameCount;
  slot->timestampMicros = (uint32_t)esp_timer_get_time();
  slot->rssiDbm = (int8_t)csiInfo->rx_ctrl.rssi;
  slot->subcarrierCount = SUBCARRIER_COUNT;
  /* buf is interleaved (imaginary, real) int8 pairs — exactly the
   * order protocol v1 uses, so a straight copy is correct.           */
  for (int byteIndex = 0; byteIndex < CSI_PAYLOAD_BYTES; byteIndex++) {
    slot->csiPairs[byteIndex] = csiInfo->buf[byteIndex];
  }
  ringWriteIndex = nextWriteIndex;
}

/* Enable CSI capture in the WiFi driver.                             */
static void enableCsiCapture() {
  wifi_csi_config_t csiConfig;
  memset(&csiConfig, 0, sizeof(csiConfig));
  csiConfig.lltf_en = true;            /* legacy long training field  */
  csiConfig.htltf_en = true;           /* HT long training field      */
  csiConfig.stbc_htltf2_en = true;     /* STBC HT-LTF2                */
  csiConfig.ltf_merge_en = true;       /* merge LTFs (less noise)     */
  csiConfig.channel_filter_en = true;  /* in-channel frames only      */
  csiConfig.manu_scale = false;        /* automatic scaling           */

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csiConfig));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&onCsiCaptured, NULL));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.print("WiSentry receiver RX-");
  Serial.print(DEVICE_ID);
  Serial.println(" starting...");

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  /* Join the sensing network. Keep trying forever with status dots:
   * a receiver that cannot join is useless, so there is no fallback. */
  WiFi.mode(WIFI_STA);
  WiFi.begin(NETWORK_SSID, NETWORK_PASSWORD);
  Serial.print("Joining '");
  Serial.print(NETWORK_SSID);
  Serial.print("'");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. My IP = ");
  Serial.print(WiFi.localIP());
  Serial.print(", RSSI = ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");

  enableCsiCapture();
  Serial.println("CSI capture enabled. Streaming to laptop.");
}

void loop() {
  /* Drain every CSI frame the callback queued and send each one as a
   * single UDP datagram (network calls belong here, not in the
   * callback).                                                       */
  while (ringReadIndex != ringWriteIndex) {
    /* Copy out of the volatile ring before touching the network.     */
    CsiDatagram outgoing;
    memcpy(&outgoing, (const void *)&ringBuffer[ringReadIndex],
           sizeof(CsiDatagram));
    ringReadIndex = (ringReadIndex + 1) % RING_SLOTS;

    csiUdpSocket.beginPacket(LAPTOP_ADDRESS, LAPTOP_UDP_PORT);
    csiUdpSocket.write((const uint8_t *)&outgoing, sizeof(CsiDatagram));
    csiUdpSocket.endPacket();
    sentDatagramCount++;
  }

  /* Once a second: LED toggle + status line for the serial monitor.  */
  unsigned long nowMillis = millis();
  if (nowMillis - lastStatusMillis >= STATUS_REPORT_INTERVAL_MS) {
    lastStatusMillis = nowMillis;
    digitalWrite(STATUS_LED_PIN,
                 digitalRead(STATUS_LED_PIN) == HIGH ? LOW : HIGH);
    Serial.print("csi captured: ");
    Serial.print(capturedFrameCount);
    Serial.print(", sent: ");
    Serial.print(sentDatagramCount);
    Serial.print(", dropped (ring full): ");
    Serial.print(droppedFrameCount);
    Serial.print(", wifi rssi: ");
    Serial.println(WiFi.RSSI());
  }

  /* If WiFi drops, reconnect — CSI capture survives reassociation.   */
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost — reconnecting...");
    WiFi.reconnect();
    delay(1000);
  }
}
