import os
from celery import shared_task
from gpuphot.image_processor import create_processor
from astropy.io import fits

INSTRUMENT_NAME = os.environ.get('INSTRUMENT_NAME', 'default_instrument')
INSTRUMENT_CONFIG_PATH = os.environ.get('INSTRUMENT_CONFIG_PATH', '/app/gpuphot/instrument_configs')
IMAGE_BASE_PATH = os.environ.get('IMAGE_BASE_PATH', '/app/images')

processor = create_processor(INSTRUMENT_NAME, INSTRUMENT_CONFIG_PATH)


@shared_task
def process_image_task(image_directory):
    image_path = os.path.join(IMAGE_BASE_PATH, image_directory)
    fits_files = [f for f in os.listdir(image_path) if f.endswith('.fits')]

    results = []
    for fits_file in fits_files:
        file_path = os.path.join(image_path, fits_file)
        with fits.open(file_path) as hdul:
            imdata = hdul[0].data
            imheader = hdul[0].header
            header_descriptions = {k: v for k, v in hdul[0].header.cards}

        dfm, original_header = processor.process_image(imdata, imheader, header_descriptions)
        results.append({
            'file': fits_file,
            'dfm': dfm,
            'header': dict(original_header)
        })

    return results


@shared_task
def get_header_info_task(image_directory):
    image_path = os.path.join(IMAGE_BASE_PATH, image_directory)
    fits_files = [f for f in os.listdir(image_path) if f.endswith('.fits')]

    results = []
    for fits_file in fits_files:
        file_path = os.path.join(image_path, fits_file)
        with fits.open(file_path) as hdul:
            header = hdul[0].header

        translated_header = processor.get_header_info(header)
        results.append({
            'file': fits_file,
            'header': dict(translated_header)
        })

    return results