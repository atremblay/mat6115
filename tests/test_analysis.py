from mat6115.analysis import unique_fixed
import numpy as np


def test_unique_fixed():

    fixed_points = np.ones((2, 10))
    assert unique_fixed(fixed_points).shape == (1, 10)

    fixed_points = np.concatenate([fixed_points, np.zeros((1, 10))], axis=0)
    unique_fixed_points = unique_fixed(fixed_points)
    assert unique_fixed_points.shape == (2, 10)
    assert (unique_fixed_points[0] == 1).all() or (unique_fixed_points[1] == 1).all()
    assert (unique_fixed_points[0] == 0).all() or (unique_fixed_points[1] == 0).all()

