import ast
import os

from gpuphot.logger.hierarchical_logging import hierarchical_debug, setup_logger

logger = setup_logger(__name__)


class ImportTransformer(ast.NodeTransformer):
    """
    Clase para transformar importaciones absolutas a importaciones relativas.
    """

    def __init__(self, module_path):
        super().__init__()
        self.module_path = os.path.abspath(module_path)
        self.project_root = self._find_project_root()
        self.module_name = self._get_module_name()

    def _find_project_root(self):
        """
        Encuentra la raíz del proyecto buscando el archivo más cercano que indique la raíz (e.g., pyproject.toml, setup.py).
        """
        current_dir = os.path.dirname(self.module_path)
        while current_dir != os.path.dirname(current_dir):
            if any((os.path.exists(os.path.join(current_dir, f)) for f in ['pyproject.toml', 'setup.py'])):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return os.path.dirname(self.module_path)

    def _get_module_name(self):
        """
        Obtiene el nombre del módulo a partir de la ruta del archivo.
        """
        module_path = os.path.relpath(self.module_path, start=self.project_root)
        return os.path.splitext(module_path)[0].replace(os.path.sep, '.')


def transform_imports_in_file(file_path):
    """
    Transforma las importaciones absolutas a importaciones relativas en un archivo Python dado.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
    transformer = ImportTransformer(file_path)
    transformed_tree = transformer.visit(tree)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(ast.unparse(transformed_tree))


def transform_imports_in_directory(directory, exclude_dirs=None):
    """
    Transforma las importaciones absolutas a importaciones relativas en todos los archivos Python dentro de un directorio.
    Excluye directorios específicos como .venv.
    """
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__']
    directory = os.path.abspath(directory)
    for (root, dirs, files) in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    transform_imports_in_file(file_path)
                    print(f'Transformado: {file_path}')
                except Exception as e:
                    print(f'Error al transformar {file_path}: {e}')


if __name__ == '__main__':
    directorio = input('Introduce el directorio que deseas analizar: ')
    exclude_dirs = ['.venv', '__pycache__', 'dev']
    transform_imports_in_directory(directorio, exclude_dirs)
