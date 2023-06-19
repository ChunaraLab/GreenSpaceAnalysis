import pytest
import numpy as np

from green_spaces.calculate_indices import apply_vegetation_index
from green_spaces.vegetation_analysis import GreenFromHSV
from tests.image_test_helpers import CountRedPixels, image_top_left_quarter_red_remainder_blue_4x4, \
    mask_top_half_true_4x4, mask_all_false_4x4, mask_all_true_4x4


@pytest.mark.parametrize("index_obj, img, mask, expected_score", [
    # 1/2 of masked pixels are red
    (CountRedPixels(), image_top_left_quarter_red_remainder_blue_4x4, mask_top_half_true_4x4, 
     (0.5, 8, [[[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
               [[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
               [[138, 138, 138], [138, 138, 138], [138, 138, 138], [138, 138, 138]],
               [[138, 138, 138], [138, 138, 138], [138, 138, 138], [138, 138, 138]]])),

    # all pixels skipped
    (CountRedPixels(), image_top_left_quarter_red_remainder_blue_4x4, mask_all_false_4x4,
     (0.0, 0, image_top_left_quarter_red_remainder_blue_4x4)),

    # 1/4 are red...
    (CountRedPixels(), image_top_left_quarter_red_remainder_blue_4x4, mask_all_true_4x4,
     (0.25, 16, [[[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
                 [[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
                 [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
                 [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]])),

    # all pixels skipped
    (GreenFromHSV({'short_name': 'HSV test', 'threshold_low': 30, 'threshold_high': 80}),
     image_top_left_quarter_red_remainder_blue_4x4, mask_all_false_4x4,
     (0.0, 0, image_top_left_quarter_red_remainder_blue_4x4)),

    # 3/4 are green...
    (GreenFromHSV({'short_name': 'HSV test', 'threshold_low': -30, 'threshold_high': 30}),
     image_top_left_quarter_red_remainder_blue_4x4, mask_all_true_4x4,
     (0.25, 16, [[[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
                 [[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
                 [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
                 [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]])),

    # 1/2 are green...
    (GreenFromHSV({'short_name': 'HSV test', 'threshold_low': -30, 'threshold_high': 30}),
     image_top_left_quarter_red_remainder_blue_4x4, mask_top_half_true_4x4,
     (0.5, 8, [[[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
               [[0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]],
               [[138, 138, 138], [138, 138, 138], [138, 138, 138], [138, 138, 138]],
               [[138, 138, 138], [138, 138, 138], [138, 138, 138], [138, 138, 138]]])),
])
def test_apply_vegetation_func(index_obj, img, mask, expected_score):
    actual_score = apply_vegetation_index(index_obj, img, mask)

    assert actual_score[0] == expected_score[0]
    assert actual_score[1] == expected_score[1]
    assert np.array_equal(actual_score[2], np.array(expected_score[2]))
