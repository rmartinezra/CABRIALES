import unittest
from pathlib import Path

import cabriales


class CabrialesCliTests(unittest.TestCase):
    def test_background_path_records_worker_count(self):
        self.assertEqual(
            cabriales.default_background_90d_out("run_test", 6),
            Path("run_test/10_in_scattering_background/")
            / "machin90d_4points_volcano_surface_step10m_workers6",
        )

    def test_background_path_rejects_zero_workers(self):
        with self.assertRaises(ValueError):
            cabriales.default_background_90d_out("run_test", 0)

    def test_full_defaults_to_ten_workers_and_dynamic_output(self):
        args = cabriales.build_parser().parse_args(["full", "--dry-run"])
        self.assertEqual(args.workers, 10)
        self.assertEqual(args.ray_step_m, 10.0)
        self.assertEqual(args.min_survival_rock_m, 10.0)
        self.assertEqual(args.kernel_energy_extrapolation, "momentum-scale")
        self.assertIsNone(args.background_out_root)


if __name__ == "__main__":
    unittest.main()
