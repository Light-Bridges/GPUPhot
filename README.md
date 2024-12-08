# GPUPhot

GPUPhot is a Python library for GPU-accelerated photometry and astrometry.

## Installation

Before installing the `gpuphot` library, ensure that the necessary system packages are installed. You can use the provided `install_dependencies.sh` script:

```bash
bash install_dependencies.sh
```

To install the `gpuphot` library, use the command:

```bash
pip install .
```

## Running with Docker Compose

To run GPUPhot using Docker Compose, follow these steps:

1. **Create a `.env` file:**

Create a file named `.env` in the same directory as your `docker-compose.yml`. This file will contain the necessary environment variables for configuration. An example `.env` file is shown below. Make sure to adjust the values according to your specific configuration:

```
INSTRUMENT_NAME=default
INSTRUMENT_CONFIG_PATH=./gpuphot/instrument_configs
IMAGE_BASE_PATH=~/gpuphot_images
CELERY_CONCURRENCY=1
```

2. **Run Docker Compose:**

Once you've created and configured the `.env` file, you can run Docker Compose with the following command:

```bash
docker-compose up -d
```

This will build the necessary Docker images (if not already built) and start the containers in detached mode (in the background).

3. **Access JupyterLab:**

You can access JupyterLab by opening your web browser and navigating to `http://localhost:8888`.

4. **Access Flower (Celery Monitor):**

You can access Flower, the Celery task monitor, at `http://localhost:5555`. This allows you to monitor the status of image processing tasks.

5. **Stop the containers:**

To stop the containers, run the following command:

```bash
docker-compose down
```

## Environment Variables

The following environment variables can be configured in the `.env` file:

* **`INSTRUMENT_NAME`**: Name of the instrument to load the configuration for. Default: `default`.
* **`INSTRUMENT_CONFIG_PATH`**: Path to the directory containing instrument configuration files. Default: `./gpuphot/instrument_configs`.
* **`IMAGE_BASE_PATH`**: Base path for storing images. Default: `~/gpuphot_images`. Inside the container, this path is mapped to `/home/jovyan/images`.
* **`CELERY_CONCURRENCY`**: Number of Celery workers to run concurrently. Adjust this value based on your GPU resources. Default: `1`.

Make sure to adjust these variables according to your specific configuration.

## Usage

To use GPUPhot in your Python scripts or Jupyter notebooks, you can import it as follows:

```python
import gpuphot
```

For detailed usage instructions and API documentation, please refer to the [official documentation](#).

## Contributing

Contributions to GPUPhot are welcome! Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.

## License

GPUPhot is released under the [MIT License](LICENSE).
