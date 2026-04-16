import os
import sys
import time
from datetime import timedelta

from dotenv import load_dotenv

# Asegúrese de que todas las rutas estén correctamente configuradas
sys.path.append('/app')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

load_dotenv(dotenv_path=os.path.join(parent_dir, '.env'))
sys.path.append(parent_dir)

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
