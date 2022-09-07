import numpy as np


def test_rng():
    seed = 100 

    rng = np.random.default_rng(seed)
    print(rng.integers(10))

    rng = np.random.default_rng(seed)
    b = np.random.randint(0, 10, 10)
    print(rng.integers(10))

test_rng()