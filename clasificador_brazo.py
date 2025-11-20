import os
import cv2
import time
import torch
import numpy as np
import threading
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pyModbusTCP.server import ModbusServer, DataBank

# ===========================
# DETECTRON2 CONFIGURATION
# ===========================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.DATASETS.TRAIN = ("prueba1",)
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  

predictor = DefaultPredictor(cfg)
CLASS_NAMES = ["COIL ON PLUG", "DISTRIBUTOR", "IGNITION COIL", "BACKGROUND"]

# ===========================
# MODBUS SERVER CONFIGURATION
# ===========================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 502
REGISTER_ADDR = 128
UPDATE_DELAY = 0.5

data_bank = DataBank()
server = ModbusServer(host=SERVER_HOST, port=SERVER_PORT, no_block=True, data_bank=data_bank)

def start_modbus_server():
    """Run Modbus server in a separate thread."""
    try:
        print(f"🔌 Starting ModbusTCP server at {SERVER_HOST}:{SERVER_PORT} ...")
        server.start()
        print("✅ Modbus server is running.")
    except Exception as e:
        print(f"❌ Modbus server error: {e}")

threading.Thread(target=start_modbus_server, daemon=True).start()

# ===========================
# REAL-TIME DETECTION
# ===========================
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("❌ Camera is not accessible.")
    exit()

cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
print("🎥 Camera running. Press 'q' to quit.")

current_value = 999
last_confirmed_value = 999
pending_class = None
start_wait_time = None
STABILITY_TIME = 5.0  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Unable to read frame. Closing...")
            break

        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            classes = instances.pred_classes.numpy()
            detected_class = int(classes[0])

            # 🔁 Map detected class to Modbus value
            if detected_class == 0:
                detected_value = 2  # COIL ON PLUG
            elif detected_class == 1:
                detected_value = 0  # DISTRIBUTOR
            elif detected_class == 2:
                detected_value = 1  # IGNITION COIL
            else:
                detected_value = 999

            class_name = CLASS_NAMES[detected_class]
            print(f"🟩 Detected: {class_name} → Modbus value {detected_value}")
        else:
            detected_value = 999
            print("🟨 No detections")

        # ===========================
        # STABILITY LOGIC (5s delay)
        # ===========================
        current_time = time.time()

        if detected_value != pending_class:
            # New detection → start stability timer
            pending_class = detected_value
            start_wait_time = current_time
            print(f"⏳ New detection: waiting {STABILITY_TIME}s to confirm...")

        elif start_wait_time is not None and (current_time - start_wait_time) >= STABILITY_TIME:
            # Class has been stable for 5s → confirm detection
            if detected_value != last_confirmed_value:
                data_bank.set_input_registers(REGISTER_ADDR, [detected_value])
                last_confirmed_value = detected_value
                print(f"✅ Value confirmed after {STABILITY_TIME}s: {detected_value}")

        # ===========================
        # VISUALIZATION
        # ===========================
        vis = Visualizer(frame[:, :, ::-1],
                         MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                         scale=1.0)
        vis = vis.draw_instance_predictions(instances)
        result_frame = vis.get_image()[:, :, ::-1]

        cv2.imshow("Live Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(UPDATE_DELAY)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    server.stop()
    print("🛑 Modbus server stopped.")
