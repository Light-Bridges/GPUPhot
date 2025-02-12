# GPUPhot

GPUPhot is a Python library for GPU-accelerated photometry and astrometry. It supports distributed processing using Celery and Docker Compose.

## Features

- Fast image processing using CUDA
- Automated astrometry and photometry
- Integration with Celery for distributed task management
- Docker Compose support for easy deployment
- JupyterLab integration for interactive data analysis

## Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

Quick install:

```
bash install.sh
pip install .
```

## Running with Docker Compose

GPUPhot can be deployed using Docker Compose. This setup includes Celery workers, RabbitMQ, Redis, PostgreSQL, JupyterLab, and Flower for task monitoring.

For detailed instructions, see [DOCKER.md](DOCKER.md).

## Usage

GPUPhot can be used in Python scripts or Jupyter notebooks. For usage examples, see [USAGE.md](USAGE.md).

Basic usage:

```
from gpuphot.image_processor import create_processor
from gpuphot_worker.utils import open_image_file

processor = create_processor('default', '/path/to/configs')
imdata, imheader = open_image_file('/path/to/image.fits')
phot_df, hwcs = processor.process_image(imdata, imheader)
print(phot_df)
```
## Important Notes

- **Instrument Configuration**: GPUPhot uses instrument-specific configuration files. See [USAGE.md](USAGE.md#instrument-configuration) for details.
- **Astrometry Index Files**: Before using the library, ensure you have downloaded the necessary astrometry index files. See [USAGE.md](USAGE.md#astrometry-setup) for instructions.

## Contributing

Contributions to GPUPhot are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

GPUPhot is released under the [MIT License](LICENSE).

[//]: # ()
[//]: # (---------------------------)

[//]: # (# OLD &#40;To delete&#41;)

[//]: # (# GPUPhot)

[//]: # ()
[//]: # (GPUPhot is a Python library for GPU-accelerated photometry and astrometry.)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (Before installing the `gpuphot` library, ensure that the necessary system packages are installed. You can use the)

[//]: # (provided `install_dependencies.sh` script:)

[//]: # ()
[//]: # (```bash)

[//]: # (bash install.sh)

[//]: # (```)

[//]: # ()
[//]: # (To install the `gpuphot` library, use the command:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install .)

[//]: # (```)

[//]: # ()
[//]: # (## Running with Docker Compose)

[//]: # ()
[//]: # (To run GPUPhot using Docker Compose, follow these steps:)

[//]: # ()
[//]: # (1. **Create a `.env` file:**)

[//]: # ()
[//]: # (Create a file named `.env` in the same directory as your `docker-compose.yml`. This file will contain the necessary)

[//]: # (environment variables for configuration. An example `.env` file is shown below. Make sure to adjust the values according)

[//]: # (to your specific configuration:)

[//]: # ()
[//]: # (```)

[//]: # (INSTRUMENT_NAME=default)

[//]: # (INSTRUMENT_CONFIG_PATH=./gpuphot/instrument_configs)

[//]: # (IMAGE_BASE_PATH=~/gpuphot_images)

[//]: # (CELERY_CONCURRENCY=1)

[//]: # (```)

[//]: # ()
[//]: # (2. **Run Docker Compose:**)

[//]: # ()
[//]: # (Once you've created and configured the `.env` file, you can run Docker Compose with the following command:)

[//]: # ()
[//]: # (```bash)

[//]: # (docker compose up -d)

[//]: # (```)

[//]: # ()
[//]: # (This will build the necessary Docker images &#40;if not already built&#41; and start the containers in detached mode &#40;in the)

[//]: # (background&#41;.)

[//]: # ()
[//]: # (3. **Access JupyterLab:**)

[//]: # ()
[//]: # (You can access JupyterLab by opening your web browser and navigating to `http://localhost:8888`.)

[//]: # ()
[//]: # (4. **Access Flower &#40;Celery Monitor&#41;:**)

[//]: # ()
[//]: # (You can access Flower, the Celery task monitor, at `http://localhost:5555`. This allows you to monitor the status of)

[//]: # (image processing tasks.)

[//]: # ()
[//]: # (5. **Stop the containers:**)

[//]: # ()
[//]: # (To stop the containers, run the following command:)

[//]: # ()
[//]: # (```bash)

[//]: # (docker compose down)

[//]: # (```)

[//]: # ()
[//]: # (## Environment Variables)

[//]: # ()
[//]: # (The following environment variables can be configured in the `.env` file:)

[//]: # ()
[//]: # (* **`INSTRUMENT_NAME`**: Name of the instrument to load the configuration for. Default: `default`.)

[//]: # (* **`INSTRUMENT_CONFIG_PATH`**: Path to the directory containing instrument configuration files. Default:)

[//]: # (  `./gpuphot/instrument_configs`.)

[//]: # (* **`IMAGE_BASE_PATH`**: Base path for storing images. Default: `~/gpuphot_images`. Inside the container, this path is)

[//]: # (  mapped to `/data/images`.)

[//]: # (* **`CELERY_CONCURRENCY`**: Number of Celery workers to run concurrently. Adjust this value based on your GPU resources.)

[//]: # (  Default: `1`.)

[//]: # ()
[//]: # (Make sure to adjust these variables according to your specific configuration.)

[//]: # ()
[//]: # (## Usage)

[//]: # ()
[//]: # (To use GPUPhot in your Python scripts or Jupyter notebooks, you can import it as follows:)

[//]: # ()
[//]: # (```python)

[//]: # (import gpuphot)

[//]: # (```)

[//]: # ()
[//]: # (For detailed usage instructions and API documentation, please refer to the [official documentation]&#40;#&#41;.)

[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (Contributions to GPUPhot are welcome! Please refer to the [contribution guidelines]&#40;CONTRIBUTING.md&#41; for more)

[//]: # (information.)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (GPUPhot is released under the [MIT License]&#40;LICENSE&#41;.)
