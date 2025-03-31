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
        print("Resultado del procesamiento:")  # Añadido para claridad
        print(result)
    except Exception as e:
        # Imprime el traceback completo para más detalles
        import traceback
        print(f"Error durante process_image_task:")
        traceback.print_exc()
        # El logger ya debería haber capturado esto, pero una impresión extra puede ayudar
        print(f"Mensaje de error resumido: {e}")
    finally:  # Asegura que el tiempo se mida incluso si hay error
        end_time = time.time()
        elapsed_time = end_time - start_time
        readable_time = str(timedelta(seconds=elapsed_time))
        print(f"Time elapsed: {readable_time}")


if __name__ == "__main__":
    main()
