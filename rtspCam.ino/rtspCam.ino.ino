#include "esp_camera.h"
#include <WiFi.h>

#include "OV2640.h"
#include "OV2640Streamer.h"
#include "CRtspSession.h"

const char* ssid = "POCO C40";
const char* password = "dannyayo";

OV2640 cam;

WiFiServer rtspServer(8554);

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("RTSP URL: rtsp://");
  Serial.print(WiFi.localIP());
  Serial.println(":8554/mjpeg/1");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_QQVGA;   // 🔥 LOW LATENCY
  config.jpeg_quality = 15;              // 🔥 COMPRESSED
  config.fb_count = 1;                  // 🔥 NO BUFFER

  esp_camera_init(&config);

  cam.init(esp_camera_fb_get, esp_camera_fb_return);

  rtspServer.begin();
}

void loop() {
  WiFiClient client = rtspServer.accept();

  if (client) {
    OV2640Streamer streamer(&client, cam);
    CRtspSession session(&client, streamer);

    while (client.connected()) {
      session.handleRequests(0);
    }
  }
}