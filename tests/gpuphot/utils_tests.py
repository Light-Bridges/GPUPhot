import os.path

from gpuphot.logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)

def get_tests_data_path():
    """
    Get the path to the test data directory.

    Returns
    -------
    str
        Path to the test data directory.
    """
    from pathlib import Path

    # Ruta del archivo actual
    current_file_path = Path(__file__).resolve()

    # Buscar la ruta que contiene "tests/gpuphot"
    project_root = current_file_path
    while not (project_root / "tests/gpuphot").exists() and project_root != project_root.root:
        project_root = project_root.parent

    return os.path.join(project_root, "tests", "data")
