import paho.mqtt.client as mqtt
import logging

class MQTTHandler:
    def __init__(self, mqtt_host="localhost", mqtt_port=1883):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.client = mqtt.Client()
        self.connected = False

    def connect_to_external_broker(self):
        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.client.loop_start()
            self.connected = True
            logging.info(f"Connected to external MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
        except Exception as e:
            logging.error(f"Failed to connect to external MQTT broker: {e}")
            self.connected = False

    def start_embedded_broker(self):
        # For simplicity, we will use a local broker if the external one is not available.
        # In a production environment, you might want to use a more robust solution.
        self.connect_to_external_broker()
        if not self.connected:
            logging.error("Failed to start embedded MQTT broker. Please check your setup.")

    def publish_message(self, topic, message):
        if not self.connected:
            self.connect_to_external_broker()
            if not self.connected:
                self.start_embedded_broker()
        if self.connected:
            try:
                self.client.publish(topic, message)
                logging.info(f"Message published to topic '{topic}': {message}")
            except Exception as e:
                logging.error(f"Failed to publish message: {e}")
        else:
            logging.error("MQTT client is not connected. Unable to publish message.")