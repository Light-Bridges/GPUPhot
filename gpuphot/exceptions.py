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
