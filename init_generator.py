import os

def create_init_files_with_imports(project_root):
    """
    Recorrer el proyecto y crear archivos __init__.py en los directorios
    que no los tengan, agregando las importaciones de todos los módulos en cada directorio.
    También se incluye una lista __all__ para controlar las importaciones globales.
    """
    for dirpath, dirnames, filenames in os.walk(project_root):
        # Ignorar carpetas ocultas como .git (o cualquier otro directorio que comience con '.')
        if any(part.startswith('.') for part in dirpath[len(project_root):].split(os.sep)):
            continue

        # Obtener todos los archivos .py del directorio actual excepto __init__.py
        py_files = [f for f in filenames if f.endswith('.py') and f != '__init__.py']

        # Añadir subdirectorios como paquetes (si contienen archivos Python)
        subdirs_with_py_files = [d for d in dirnames if any(f.endswith('.py') for f in os.listdir(os.path.join(dirpath, d)))]

        # Si no hay archivos Python y no hay subdirectorios con .py, no es necesario un __init__.py
        if not py_files and not subdirs_with_py_files:
            continue

        # Ruta para el archivo __init__.py
        init_file = os.path.join(dirpath, '__init__.py')

        # Crear el contenido para __init__.py
        imports = []
        all_list = []

        # Añadir importaciones de módulos
        for py_file in py_files:
            module_name = py_file[:-3]  # Quitar la extensión .py
            imports.append(f'from . import {module_name}')
            all_list.append(f"'{module_name}'")

        # Añadir importaciones de subdirectorios como paquetes
        for subdir in subdirs_with_py_files:
            imports.append(f'from . import {subdir}')
            all_list.append(f"'{subdir}'")

        init_content = '\n'.join(imports) + '\n\n' + f'__all__ = [{", ".join(all_list)}]'

        # Crear o sobrescribir el archivo __init__.py con el contenido generado
        print(f'Creando {init_file} con el contenido:\n{init_content}\n')
        # with open(init_file, 'w') as f:
        #     f.write(init_content)

if __name__ == "__main__":
    # Obtén la ruta del directorio actual donde está ubicado el script
    project_root = os.path.dirname(os.path.abspath(__file__))
    create_init_files_with_imports(project_root)
