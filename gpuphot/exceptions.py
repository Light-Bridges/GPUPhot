from .logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


class GPUPhotError(Exception):
    """Base class for exceptions in the gpuphot module."""
    pass


class InsufficientStarsError(GPUPhotError):
    """Exception raised when there are not enough isolated stars for processing."""

    def __init__(self, num_stars, message="Insufficient number of isolated stars detected"):
        self.num_stars = num_stars
        self.message = f"{message}. Found {num_stars} stars, but at least 5 are required."
        super().__init__(self.message)


class MoffatFitError(GPUPhotError):
    """Exception raised when the PSF fitting is impossible."""

    def __init__(self, message="Impossible to fit Moffat function to reference PSF"):
        self.message = f"{message}"
        super().__init__(self.message)


class ImageQualityError(GPUPhotError):
    """Exception raised when the image quality is too poor for processing."""

    def __init__(self, message="Image may be too crowded or too noisy"):
        self.message = message
        super().__init__(self.message)


def capture_cuda_exception(func):
    """Decorator to capture CUDA exceptions and retry on illegal address error."""

    import os

    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    def is_running_in_docker():
        return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')

    def wrapper(*args, **kwargs):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except CUDARuntimeError as e:
                from .utils.gpu import free_gpu_mem
                free_gpu_mem()  # Liberar memoria GPU antes de manejar el error

                if "cudaErrorIllegalAddress" in str(e):
                    if is_running_in_docker():
                        logger.critical("Exiting due to CUDA error in Docker container.")
                        sys.exit('Exiting due to CUDA error.')
                    else:
                        # Solo mostrar advertencia si no es el último intento
                        if attempt < max_retries - 1:
                            logger.warning("CUDA error detected, but not running in Docker. Retrying...")
                        else:
                            logger.error("Max retries reached. Exiting due to CUDA error.")
                else:
                    # Solo mostrar advertencia si no es el último intento
                    if attempt < max_retries - 1:
                        logger.warning(f"A non-critical CUDA error occurred in {func.__name__}: {e}. Retrying...")
                    else:
                        logger.error(f"Max retries reached for {func.__name__}. Exiting.")
                        return None
                    continue

            except Exception as e:
                logger.error(f"An unexpected error occurred in {func.__name__}: {e}")
                return None

        logger.debug(f"Max retries reached for {func.__name__}. Returning None.")
        return None

    return wrapper


if __name__ == '__main__':
    import sys
    from cupy_backends.cuda.api.runtime import CUDARuntimeError


    @capture_cuda_exception
    def compute_sum():

        raise CUDARuntimeError(700)


    if compute_sum() is None:
        print("An error occurred during computation.")
        sys.exit(1)
