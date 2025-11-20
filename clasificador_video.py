from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import sys
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ==========================
# CONFIGURACIÓN DEL MODELO
# ==========================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TRAIN = ("prueba1",)
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # según entrenamiento

predictor = DefaultPredictor(cfg)

CLASS_NAMES = ["COIL ON PLUG", "DISTRIBUTOR", "IGNITION COIL", "BACKGROUND"]

# ==========================
# CAPTURA DE VIDEO EN VIVO
# ==========================
cap = cv2.VideoCapture(2)  

if not cap.isOpened():
    print("❌ No se puede abrir la cámara. Prueba otro índice (0, 1, 2).")
    sys.exit()

cv2.namedWindow("DETECTRON2 Live", cv2.WINDOW_NORMAL)  # ✅ Solo una ventana
print("✅ Cámara iniciada. Presiona 'q' para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar frame. Saliendo...")
            break

        # Realiza la predicción
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        # Dibuja las detecciones si hay alguna
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()

            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                label = CLASS_NAMES[cls]
                confidence = f"{score * 100:.1f}%"

                # Rectángulo verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Texto de clase + confianza
                text = f"{label}: {confidence}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Mostrar también en consola
                print(f"🟩 Detectado: {label} ({confidence})")

        else:
            print("🟨 Sin detecciones")

        # ✅ Mostrar en una sola ventana (siempre con el mismo nombre)
        cv2.imshow("DETECTRON2 Live", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrumpido por el usuario.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Cámara detenida correctamente.")
