from pyModbusTCP.server import ModbusServer
import time

# === Configuración ===
SERVER_HOST = "0.0.0.0"       # Escucha en todas las interfaces (incluida la red del UR)
SERVER_PORT = 502             # Puerto TCP Modbus estándar
REGISTER_ADDR = 128           # Dirección del registro
VALUE = 0                     # Valor que mantendrá el servidor
UPDATE_DELAY = 1              # Tiempo entre actualizaciones (segundos)

# Crear servidor Modbus TCP
server = ModbusServer(host=SERVER_HOST, port=SERVER_PORT, no_block=True)

try:
    print(f"🔌 Iniciando servidor ModbusTCP en {SERVER_HOST}:{SERVER_PORT} ...")
    server.start()
    print("✅ Servidor iniciado correctamente.")
    print(f"📡 Manteniendo HR[{REGISTER_ADDR}] = {VALUE}")
    print("🕹 Esperando conexión del UR30...")

    while True:
        # Mantiene actualizado el valor del holding register
        server.data_bank.set_holding_registers(REGISTER_ADDR, [VALUE])
        print(f"HR[{REGISTER_ADDR}] = {VALUE} (servidor activo)", end="\r")
        time.sleep(UPDATE_DELAY)

except Exception as e:
    print(f"\n❌ Error en el servidor: {e}")

finally:
    server.stop()
    print("\n🛑 Servidor detenido.")
