def test_cupy():
    try:
        import cupy as cp

        print("## Información de GPU con cupy:")
        print(f"Número de GPUs disponibles: {cp.cuda.runtime.getDeviceCount()}")

        for i in range(cp.cuda.runtime.getDeviceCount()):
            device = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}:")
            print(f"  Nombre: {device['name'].decode('utf-8')}")
            print(f"  Memoria total: {device['totalGlobalMem'] / (1024 ** 3):.2f} GB")
            print(f"  Capacidad de cómputo: {device['computeCapabilityMajor']}.{device['computeCapabilityMinor']}")
            print(f"  Núcleos CUDA: {device['multiProcessorCount'] * 64}")

        # Operación básica con cupy
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = cp.add(a, b)
        print("\nOperación básica con cupy:")
        print(f"  {a} + {b} = {c}")

        return True
    except ImportError:
        print("cupy no está instalado o no se puede importar.")
        return False
    except Exception as e:
        print(f"Error al usar cupy: {str(e)}")
        return False


def test_tensorflow():
    try:
        import tensorflow as tf

        print("\n## Información de GPU con TensorFlow:")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Número de GPUs disponibles: {len(gpus)}")

        # Operación básica con TensorFlow
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print("\nOperación básica con TensorFlow:")
        print(f"  {a.numpy()} + {b.numpy()} = {c.numpy()}")

        return True
    except ImportError:
        print("TensorFlow no está instalado o no se puede importar.")
        return False
    except Exception as e:
        print(f"Error al usar TensorFlow: {str(e)}")
        return False


def main():
    print("Iniciando pruebas de GPU...\n")

    cupy_success = test_cupy()
    tensorflow_success = test_tensorflow()

    print("\nResumen de pruebas:")
    print(f"cupy: {'Éxito' if cupy_success else 'Fallo'}")
    print(f"TensorFlow: {'Éxito' if tensorflow_success else 'Fallo'}")


if __name__ == "__main__":
    main()
