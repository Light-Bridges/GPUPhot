import sys
import time
from datetime import timedelta
import os
import shutil
import tempfile
from types import MethodType

# --- INICIO DEL PARCHE ---
# Monkeypatch para resolver el error de symbolic links en tempfile.TemporaryDirectory
# Asegúrate de que este código se ejecuta ANTES de cualquier importación que pueda usar TemporaryDirectory

def _safe_rmtree(cls, name, ignore_errors=False):
    """Custom _rmtree que maneja enlaces simbólicos correctamente"""
    print(f"DEBUG: Intentando limpiar '{name}' con _safe_rmtree") # Añade un print para depurar

    def onerror(func, path, exc_info):
        # Podrías añadir logging aquí también
        print(f"DEBUG: Error en onerror durante la limpieza de '{path}': {exc_info[1]}")
        if not ignore_errors:
            raise

    try:
        if os.path.islink(name):
            print(f"DEBUG: '{name}' es un enlace simbólico, usando os.unlink()")
            os.unlink(name)
        else:
            print(f"DEBUG: '{name}' no es un enlace simbólico, usando shutil.rmtree()")
            shutil.rmtree(name, ignore_errors=ignore_errors, onerror=onerror if not ignore_errors else None) # Pasa ignore_errors y onerror correctamente
    except FileNotFoundError:
         print(f"DEBUG: El archivo o directorio '{name}' no fue encontrado durante la limpieza (puede que ya haya sido borrado).")
    except Exception as e:
        print(f"DEBUG: Excepción inesperada en _safe_rmtree para '{name}': {e}")
        if not ignore_errors:
             raise # Relanza la excepción si no debemos ignorar errores

# Aplicar el parche a TemporaryDirectory
# Usamos getattr/hasattr por si _rmtree no existe en alguna versión/plataforma, aunque es estándar.
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    # Guardamos una referencia al original por si acaso (opcional)
    # tempfile.TemporaryDirectory._rmtree_original = tempfile.TemporaryDirectory._rmtree
    tempfile.TemporaryDirectory._rmtree = MethodType(_safe_rmtree, tempfile.TemporaryDirectory)
    print("DEBUG: Monkey patch para tempfile.TemporaryDirectory._rmtree aplicado.")
else:
    print("DEBUG WARNING: tempfile.TemporaryDirectory no tiene el método _rmtree para parchear.")

# --- FIN DEL PARCHE ---


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
        print("Resultado del procesamiento:") # Añadido para claridad
        print(result)
    except Exception as e:
        # Imprime el traceback completo para más detalles
        import traceback
        print(f"Error durante process_image_task:")
        traceback.print_exc()
        # El logger ya debería haber capturado esto, pero una impresión extra puede ayudar
        print(f"Mensaje de error resumido: {e}")
    finally: # Asegura que el tiempo se mida incluso si hay error
        end_time = time.time()
        elapsed_time = end_time - start_time
        readable_time = str(timedelta(seconds=elapsed_time))
        print(f"Time elapsed: {readable_time}")


if __name__ == "__main__":
    main()