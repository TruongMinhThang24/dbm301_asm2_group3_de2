import unittest

import numpy as np

from groceries_group3.common.paths import PROJECT_ROOT, RAW_DATA_DIR
from groceries_group3.common.reproducibility import seed_everything


class ProjectSetupTests(unittest.TestCase):
    def test_project_paths_point_to_repository(self) -> None:
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())
        self.assertEqual(RAW_DATA_DIR, PROJECT_ROOT / "data" / "raw")

    def test_seed_repeats_numpy_random_values(self) -> None:
        seed_everything(123)
        first = np.random.random()
        seed_everything(123)
        second = np.random.random()
        self.assertEqual(first, second)
