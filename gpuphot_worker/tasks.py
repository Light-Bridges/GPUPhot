import os
from glob import glob

from astropy.io import fits
from celery import shared_task

from gpuphot.image_processor import create_processor


def get_processor(instrument_name=None):
    """
    Create a processor with optional custom instrument and configuration path.

    Args:
        instrument_name (str, optional): Name of the instrument to use.

    Returns:
        processor: Configured image processor
    """
    instrument_name = instrument_name or os.environ.get('INSTRUMENT_NAME', 'default_instrument')
    config_base_path = os.environ.get('INSTRUMENT_CONFIG_BASE_PATH', '/app/gpuphot/instrument_configs')
    return create_processor(instrument_name, config_base_path)


@shared_task
def process_image_task(path=None, filename=None, instrument_name=None):
    """
    Process astronomical images with flexible search and configuration options.

    Task Usage Examples:
    1. Process all images in default path:
       process_image_task.delay()

    2. Process images in a specific directory:
       process_image_task.delay(path='today')

    3. Process a specific image file:
       process_image_task.delay(path='today/camera1', filename='image.fits')

    4. Process images containing a specific name pattern:
       process_image_task.delay(filename='NEO')

    5. Process with custom instrument configuration:
       process_image_task.delay(
           path='today',
           instrument_name='other_instrument'
       )

    Args:
        path (str, optional): Subdirectory to search for images.
                               If None, searches in base image path.
        filename (str, optional): Specific filename or pattern to match.
                                  Supports partial matches and wildcards.
        instrument_name (str, optional): Override default instrument name.

    Returns:
        list: Processed image results containing file details, processed data, and headers.
    """
    base_path = os.environ.get('IMAGE_BASE_PATH', '/app/images')
    processor = get_processor(instrument_name)

    if path:
        search_path = os.path.join(base_path, path)
    else:
        search_path = base_path

    if filename:
        if '*' not in filename:
            filename = f'*{filename}*'
        search_pattern = os.path.join(search_path, '**', filename)
    else:
        search_pattern = os.path.join(search_path, '**', '*.fits')

    fits_files = glob(search_pattern, recursive=True)

    results = []
    for file_path in fits_files:
        with fits.open(file_path) as hdul:
            imdata = hdul[0].data
            imheader = hdul[0].header
            header_descriptions = {k: v for k, v in hdul[0].header.cards}

        dfm, original_header = processor.process_image(imdata, imheader, header_descriptions)
        results.append({
            'file': os.path.relpath(file_path, base_path),
            'dfm': dfm,
            'header': dict(original_header)
        })

    return results
