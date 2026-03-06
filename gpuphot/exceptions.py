import nvtx

from .logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


class GPUPhotError(Exception):
    """Base class for exceptions in the gpuphot module."""
    pass


class InsufficientStarsError(GPUPhotError):
    """
    Exception raised when there are not enough isolated stars for processing.

    :param num_stars: The number of stars detected.
    :type num_stars: int
    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, num_stars, message="Insufficient number of isolated stars detected"):
        self.num_stars = num_stars
        self.message = f"{message}. Found {num_stars} stars, but at least 5 are required."
        super().__init__(self.message)


class MoffatFitError(GPUPhotError):
    """
    Exception raised when the PSF fitting is impossible.

    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, message="Impossible to fit Moffat function to reference PSF"):
        self.message = f"{message}"
        super().__init__(self.message)


class ImageQualityError(GPUPhotError):
    """
    Exception raised when the image quality is too poor for processing.

    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, message="Image may be too crowded or too noisy"):
        self.message = message
        super().__init__(self.message)


class UnableToAstrometrizeError(GPUPhotError):
    """
    Exception raised when the astrometry process fails due to inability to astrometrize.

    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, message="Unable to astrometrize the image"):
        self.message = message
        super().__init__(self.message)


class AstrometrizationTimeoutError(GPUPhotError):
    """
    Exception raised when the astrometrization process times out.

    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, message="Astrometrization process has timed out"):
        self.message = message
        super().__init__(self.message)


class DataValidationError(GPUPhotError):
    """
    Exception raised when input data is invalid or insufficient.

    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, message="Input data is invalid or insufficient"):
        self.message = message
        super().__init__(self.message)


class InvalidGroupSizeError(GPUPhotError):
    """
    Exception raised when the average or minimum group size is invalid.

    :param avg_group_size: The average group size.
    :type avg_group_size: float
    :param min_group_size: The minimum group size.
    :type min_group_size: float
    :param message: The error message (optional).
    :type message: str
    """

    def __init__(self, avg_group_size, min_group_size, message="Invalid group size parameters"):
        self.avg_group_size = avg_group_size
        self.min_group_size = min_group_size
        self.message = f"{message}. avg_group_size: {avg_group_size}, min_group_size: {min_group_size}."
        super().__init__(self.message)


def capture_cuda_exception(func):
    """
    Decorator to capture CUDA exceptions and retry on illegal address error.

    :param func: The function to be decorated.
    :type func: callable
    :return: The wrapped function.
    :rtype: callable
    """
    import os
    try:
        # Try importing from the public API first (CuPy 10+)
        from cupy.cuda.runtime import CUDARuntimeError
    except ImportError:
        # Fallback for older versions or internal paths
        from cupy_backends.cuda.api.runtime import CUDARuntimeError

    def is_running_in_docker():
        return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')

    def wrapper(*args, **kwargs):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except CUDARuntimeError as e:
                error_str = str(e)
                nvtx.mark(f"CUDA error in {func.__name__}: {e}", color="red", category="error")
                
                # Check for critical initialization error to avoid further CUDA calls
                is_init_error = "cudaErrorInitializationError" in error_str
                
                if not is_init_error:
                    try:
                        from .utils.gpu import free_gpu_mem
                        free_gpu_mem()  # Liberar memoria GPU antes de manejar el error
                    except Exception:
                        pass

                if any(error in error_str for error in
                       ("cudaErrorIllegalAddress", "cudaErrorInitializationError", "cudaErrorInvalidValue")):
                    if is_running_in_docker():
                        logger.critical("Exiting due to CUDA error in Docker container.")
                        try:
                            # sys.exit('Exiting due to CUDA error.')
                            import socket
                            import docker

                            # Obtener hostname (que suele ser el container_id)
                            container_id = socket.gethostname()

                            # Reiniciar el contenedor
                            client = docker.DockerClient(base_url='unix://run/docker.sock')
                            client.containers.get(container_id).restart()
                        except Exception as e:
                            logger.error(f"Error restarting container: {e}")
                        raise
                    else:
                        logger.error("Max retries reached. Exiting due to CUDA error.")
                        raise  # Permitir que se propague el error para manejarlo más arriba
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"CUDA error detected: {e}. Retrying...")
                        continue  # Intentar nuevamente si hay más reintentos
                    else:
                        logger.error("Max retries reached for non-critical CUDA error. Raising exception.")
                        raise  # Lanzar excepción si se han agotado los reintentos

            except Exception as e:
                logger.error(f"An unexpected error occurred in {func.__name__}: {e}", exc_info=True)
                raise  # Permitir que se propague cualquier otro tipo de excepción

        logger.debug(f"Max retries reached for {func.__name__}.")
        return None  # Esto puede ser opcional dependiendo de cómo quieras manejar los retornos

    return wrapper


if __name__ == '__main__':
    import sys
    try:
        from cupy.cuda.runtime import CUDARuntimeError
    except ImportError:
        from cupy_backends.cuda.api.runtime import CUDARuntimeError


    @capture_cuda_exception
    def compute_sum():

        raise CUDARuntimeError(700)


    if compute_sum() is None:
        print("An error occurred during computation.")
        sys.exit(1)
