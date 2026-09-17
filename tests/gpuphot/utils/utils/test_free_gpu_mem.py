import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestFreeGpuMem(unittest.TestCase):

    def test_runs_without_error(self):
        from gpuphot.utils.gpu import free_gpu_mem
        free_gpu_mem()

    def test_can_call_multiple_times(self):
        from gpuphot.utils.gpu import free_gpu_mem
        free_gpu_mem()
        free_gpu_mem()
        free_gpu_mem()

    def test_frees_allocated_memory(self):
        from gpuphot.utils.gpu import free_gpu_mem
        # Allocate some memory
        _ = cp.ones((1000, 1000), dtype=cp.float64)
        del _
        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes()
        free_gpu_mem()
        used_after = mempool.used_bytes()
        self.assertLessEqual(used_after, used_before)

    def test_pool_bytes_after_free(self):
        from gpuphot.utils.gpu import free_gpu_mem
        _ = cp.ones((500, 500), dtype=cp.float32)
        del _
        free_gpu_mem()
        mempool = cp.get_default_memory_pool()
        self.assertEqual(mempool.used_bytes(), 0)


if __name__ == '__main__':
    unittest.main()
