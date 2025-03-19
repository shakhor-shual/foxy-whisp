import logging
from mqtt_handler import MQTTHandler

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create an instance of MQTTHandler
mqtt_handler = MQTTHandler()

# Test connecting to an external broker
mqtt_handler.connect_to_external_broker()

# If not connected, start an embedded broker
if not mqtt_handler.connected:
    mqtt_handler.start_embedded_broker()

# Test publishing a message
if mqtt_handler.connected:
    mqtt_handler.publish_message("foxy-whisp", "<foxy:started>")
else:
    logging.error("MQTT client is not connected. Unable to publish message.")