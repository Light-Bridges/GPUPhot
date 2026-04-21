import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestFillImage(unittest.TestCase):

    def test_power_of_2_unchanged(self):
        from .....gpuphot.phot.conv import fill_image
        h, w = fill_image((256, 512))
        self.assertEqual(h, 256)
        self.assertEqual(w, 512)

    def test_rounds_up(self):
        from .....gpuphot.phot.conv import fill_image
        h, w = fill_image((300, 500))
        self.assertEqual(h, 512)
        self.assertEqual(w, 512)

    def test_small_shape(self):
        from .....gpuphot.phot.conv import fill_image
        h, w = fill_image((3, 5))
        self.assertEqual(h, 4)
        self.assertEqual(w, 8)

    def test_returns_tuple(self):
        from .....gpuphot.phot.conv import fill_image
        result = fill_image((100, 200))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_result_is_power_of_2(self):
        from .....gpuphot.phot.conv import fill_image
        import math
        h, w = fill_image((1000, 2000))
        self.assertEqual(h, 2 ** int(math.ceil(math.log2(1000))))
        self.assertEqual(w, 2 ** int(math.ceil(math.log2(2000))))

    def test_one_by_one(self):
        from .....gpuphot.phot.conv import fill_image
        h, w = fill_image((1, 1))
        self.assertEqual(h, 1)
        self.assertEqual(w, 1)


if __name__ == '__main__':
    unittest.main()
