import os
import sys
import time
from datetime import timedelta

from dotenv import load_dotenv

sys.path.append('/app')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

load_dotenv(dotenv_path=os.path.join(parent_dir, '.env'))
sys.path.append(parent_dir)

from gpuphot_worker.utils import get_processor, open_image_file, BASE_IMAGES_PATH
from gpuphot.utils.gpu import reset_cupy_allocators
from gpuphot_worker.header_descriptions import HEADER_DESCRIPTIONS


def main():
    if len(sys.argv) < 2:
        print("Uso: python profile_vizier.py <ruta_imagen> [instrument_name]")
        sys.exit(1)

    file_path = sys.argv[1].lstrip('/')
    instrument_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('INSTRUMENT_NAME', 'default')

    full_path = os.path.join(BASE_IMAGES_PATH, file_path)

    print(f"Procesando imagen: {file_path} con instrumento: {instrument_name} (Vizier remoto)", flush=True)

    processor = get_processor(instrument_name)
    reset_cupy_allocators()

    start_time = time.time()
    try:
        imdata, imheader = open_image_file(full_path)
        phot_df, hwcs = processor.process_image(
            imdata, imheader,
            header_descriptions=HEADER_DESCRIPTIONS,
        )

        n_obj = len(phot_df) if phot_df is not None else 0
        n_trans = int(phot_df['trans'].sum()) if phot_df is not None and 'trans' in phot_df.columns else 0

        result = {
            'input_file': file_path,
            'imaphot': {'objets': n_obj, 'transients': n_trans, 'stored': False},
            'imastats': {'stored': False},
        }
        print("Resultado del procesamiento:", flush=True)
        print(result)
    except Exception as e:
        import traceback
        print("Error durante el procesamiento:", flush=True)
        traceback.print_exc()
        print(f"Mensaje de error resumido: {e}", flush=True)
    finally:
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {str(timedelta(seconds=elapsed_time))}", flush=True)


if __name__ == "__main__":
    main()
