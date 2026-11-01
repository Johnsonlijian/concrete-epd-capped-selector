import unittest

import numpy as np

from quantity_model import AS_IN2, PU_KIP, axial_nominal_kip, external_benchmark, net_concrete_area, saturation_strength


class QuantityModelTests(unittest.TestCase):
    def test_external_section_values(self):
        self.assertEqual(len(external_benchmark()), 14)

    def test_benchmark_capacity(self):
        self.assertAlmostEqual(axial_nominal_kip(5000, 256), 1534)
        self.assertAlmostEqual(PU_KIP, 797.68)

    def test_inverse_strength_before_floor(self):
        values = net_concrete_area(np.array([2500, 3000, 4000]), 12)
        np.testing.assert_allclose(values * [2500, 3000, 4000], [1240000] * 3)

    def test_saturation_at_reference_width(self):
        self.assertAlmostEqual(saturation_strength(16), 5000)
        np.testing.assert_allclose(net_concrete_area([5000, 6000, 8000], 16), [256 - AS_IN2] * 3)

    def test_capped_index_identity(self):
        strengths = np.linspace(2500, 8000, 111)
        for width in (12, 14, 16, 18):
            fstar = saturation_strength(width)
            areas = net_concrete_area(strengths, width)
            np.testing.assert_allclose(areas, 1240000 / np.minimum(strengths, fstar))


if __name__ == "__main__":
    unittest.main()
