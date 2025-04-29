import cupy as cp
from functools import wraps
import inspect

FUNCIONES_CRITICAS = ['array', 'zeros', 'empty', 'zeros_like', 'empty_like']


def apply_float64_patch():
    _originals = {}

    def wrap_array_creator(original):
        # Analizar firma de la función original
        sig = inspect.signature(original)
        params = list(sig.parameters.values())

        # Encontrar posición del parámetro 'dtype'
        dtype_pos = None
        for i, p in enumerate(params):
            if p.name == 'dtype':
                dtype_pos = i
                break

        @wraps(original)
        def wrapped(*args, **kwargs):
            # Determinar si 'dtype' está siendo provisto posicionalmente
            dtype_provided = (
                    (dtype_pos is not None and len(args) > dtype_pos) or  # Posicional
                    'dtype' in kwargs  # Keyword
            )

            # Añadir dtype=cp.float64 solo si no está presente
            if not dtype_provided:
                kwargs['dtype'] = cp.float64

            return original(*args, **kwargs)

        return wrapped

    for func_name in FUNCIONES_CRITICAS:
        original = getattr(cp, func_name)
        _originals[func_name] = original
        setattr(cp, func_name, wrap_array_creator(original))

    return _originals


# Aplicar el parche al importar
_original_cupy_funcs = apply_float64_patch()
