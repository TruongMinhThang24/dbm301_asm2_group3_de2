import random

import numpy as np


RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
