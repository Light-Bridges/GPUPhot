import os
import sys
import time
from datetime import timedelta

# Asegúrese de que todas las rutas estén correctamente configuradas
sys.path.append('/app')

# Importe su función principal DESPUÉS del parche
from gpuphot_worker.tasks import process_image_task


def main():
    if len(sys.argv) < 2:
        print("Uso: python profile_image_processing.py <ruta_imagen> [instrument_name]")
        sys.exit(1)

    file_path = sys.argv[1]
    instrument_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('INSTRUMENT_NAME', 'default')

    print(f"Procesando imagen: {file_path} con instrumento: {instrument_name}")

    start_time = time.time()
    try:
        result = process_image_task(file_path, instrument_name=instrument_name)
        print("Resultado del procesamiento:", flush=True)
        print(result)
    except Exception as e:
        import traceback
        print(f"Error durante process_image_task:", flush=True)
        traceback.print_exc()
        print(f"Mensaje de error resumido: {e}", flush=True)
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        readable_time = str(timedelta(seconds=elapsed_time))
        print(f"Time elapsed: {readable_time}", flush=True)


if __name__ == "__main__":
    main()
