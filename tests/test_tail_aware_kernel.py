from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from modulos.empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "modulos/hybrid_empirical_kernel_library.npz"


class TailAwareKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = TailAwareEmpiricalKernel(MODEL)
        cls.prediction = cls.model.predict_kernel(80.0, 39.67)

    def test_hybrid_loader_prefers_full_tail(self) -> None:
        library = load_empirical_kernel_library(MODEL)
        self.assertEqual(library.family, "full_tail")
        self.assertEqual(library.centers_mrad.size, 3200)
        self.assertEqual(self.model.transport_L_min_m, 1.0)
        self.assertEqual(self.model.transport_L_max_m, 1500.0)

    def test_prediction_is_normalized_symmetric_and_nonnegative(self) -> None:
        probability = self.prediction.probability_per_bin
        self.assertTrue(self.prediction.valid)
        self.assertEqual(probability.size, 3200)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)
        self.assertTrue(np.all(probability >= 0.0))
        np.testing.assert_allclose(probability, probability[::-1], rtol=0.0, atol=1e-15)

    def test_hard_scattering_tail_is_preserved(self) -> None:
        centers = self.prediction.centers_mrad
        probability = self.prediction.probability_per_bin
        self.assertGreater(float(probability[np.abs(centers) > 300.0].sum()), 1e-3)
        self.assertGreater(float(probability[np.abs(centers) > 500.0].sum()), 1e-5)
        self.assertGreater(float(probability[np.abs(centers) > 1000.0].sum()), 1e-8)
        self.assertEqual(self.prediction.tail_policy, "body_quantile_tail_histogram_linear")


if __name__ == "__main__":
    unittest.main()
