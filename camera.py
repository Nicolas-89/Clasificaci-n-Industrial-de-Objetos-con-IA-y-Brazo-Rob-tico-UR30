import cv2
import os
import re

# === Ruta donde se guardarán las imágenes ===
save_path = os.path.expanduser("images")
os.makedirs(save_path, exist_ok=True)

# === Inicializar cámara ===
cap = cv2.VideoCapture(2)

# === Buscar el último número de archivo guardado ===
existing_images = [f for f in os.listdir(save_path) if f.lower().endswith(".jpg")]

# Extraer números de los nombres (ej: "12.jpg" -> 12)
numbers = []
for f in existing_images:
    match = re.search(r'(\d+)\.jpg$', f)
    if match:
        numbers.append(int(match.group(1)))

# Si hay imágenes, empieza desde el siguiente número; si no, desde 1
image_number = max(numbers) + 1 if numbers else 1

print(f"📸 Empezando desde la imagen número {image_number}.")
print("Presiona ESPACIO para tomar una foto o ESC para salir.")

# === Bucle principal ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar el cuadro.")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    
    # Tomar foto (tecla ESPACIO)
    if key == 32:  # SPACE
        img_name = f"{image_number}.jpg"
        img_path = os.path.join(save_path, img_name)
        cv2.imwrite(img_path, frame)
        print(f"✅ Guardado: {img_path}")
        image_number += 1

    # Salir (tecla ESC)
    elif key == 27:  # ESC
        print("👋 Cerrando cámara.")
        break

# === Liberar recursos ===
cap.release()
cv2.destroyAllWindows()
