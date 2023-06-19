import platform
import timeit

import numpy as np

from green_spaces.vegetation_analysis import GreenFromHSV, GreenLeafIndex, NormalizedDifferenceVegetationIndexCIR, \
    VisualNormalizedDifferenceVegetationIndex, VisualAtmosphericResistanceIndex, GreenFromLab1, GreenFromLab2, \
    AssumesGreen, MattIrHSV
from tests.image_test_helpers import image_left_half_red_right_half_blue_4x4

visual_confirmation = True


def test_hsv_threshold_func():
    expected_indices = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    # blue has hie=24-; hsv space in cv2 has range 0-180 so need to accept hue = 240/2 = 120
    hsv_threshold = GreenFromHSV({'short_name': 'HSV test', 'threshold_low': 105, 'threshold_high': 135})
    actual_indices = hsv_threshold.index(image_left_half_red_right_half_blue_4x4)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_green_leaf_index_func():
    # gli = ((green - red) + (green - blue)) / ((2.0 * green) + red + blue)
    test_image = np.array([  # B G R
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [0, 255, 255], [255, 0, 255]]
    ], dtype=np.uint8)
    # green leaf values = [[-1, 1, -1], [1.0/3.0, 1.0/3.0, -1]]
    expected_indices = np.array([[0, 0, 0], [1, 1, 0]])
    green_leaf = GreenLeafIndex({'short_name': 'GLI test', 'threshold_low': 0.2, 'threshold_high': 0.8})
    actual_indices = green_leaf.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_normalized_difference_vegetation_index_func():
    # ndvi = (nir - red) / (nir + red)
    # 1 if 0.2 <= ndvi <= 0.8
    test_image_grir = np.array([  # iR G R
        [[0, 0, 255], [255, 0, 0], [0, 255, 0]],
        [[0, 128, 255], [0, 255, 128], [0, 255, 255]]
    ], dtype=np.uint8)
    # ndvi value = [ [1, NaN, -1], [0.333, -0.333, 0]]
    expected_indices = np.array([[0, 0, 0], [1, 0, 0]])
    ndvi = NormalizedDifferenceVegetationIndexCIR({'short_name': 'NDVI test',
                                                   'threshold_low': 0.2, 'threshold_high': 0.8})
    actual_indices = ndvi.index(test_image_grir)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_visual_normalized_difference_vegetation_index_func():
    # vndvi = (green - red) / (green + red)
    # 1 if 0.2 <= ndvi <= 0.8
    test_image = np.array([  # B G R
        [[0, 0, 255], [0, 0, 0], [0, 255, 0]],
        [[0, 128, 255], [0, 255, 128], [0, 255, 255]]
    ], dtype=np.uint8)
    # vndvi value = [ [-1, NaN, 1], [-0.333, 0.333, 0]]
    expected_indices = np.array([[0, 0, 0], [0, 1, 0]])
    vndvi = VisualNormalizedDifferenceVegetationIndex({'short_name': 'vNDVI test',
                                                       'threshold_low': 0.2, 'threshold_high': 0.8})
    actual_indices = vndvi.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_visual_atmospheric_resistance_index_func():
    # vari = (green - red) / (green + red + blue)
    # 1 if 0.2 <= ndvi <= 0.8
    test_image = np.array([  # B G R
        [[0, 0, 255], [0, 0, 0], [0, 255, 0]],
        [[0, 128, 255], [0, 255, 128], [0, 255, 255]],
        [[250, 255, 128], [255, 255, 128], [255, 255, 0]],
    ], dtype=np.uint8)
    # vndvi value = [ [-1, NaN 1], [-0.333, 0.333, 0], [0.200, 0.199, 0.5]]
    expected_indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]])
    vari = VisualAtmosphericResistanceIndex({'short_name': 'VARI test', 'threshold_low': 0.2, 'threshold_high': 0.8})
    actual_indices = vari.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_green_from_lab1_index_func():
    # lab1 = L, a, b from R,G,B
    # 1 if -9999 <= a <= -11
    test_image = np.array([  # B G R
        [[118, 125, 93], [119, 124, 100]],
    ], dtype=np.uint8)
    # a value = [ [-12.97, -9.79] ]
    expected_indices = np.array([[1, 0]])
    lab1 = GreenFromLab1({'short_name': 'L*a*b* v1 test', 'a_threshold_low': -9999, 'a_threshold_high': -11})
    actual_indices = lab1.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_green_from_lab12_index_func():
    # lab1 = L, a, b from R,G,B
    # 1 if -9999 <= a <= -6 and 5 <= b <= 57
    test_image = np.array([  # B G R
        [[113, 125, 96], [67, 124, 114]],
        [[178, 97, 69], [153, 101, 153]],
    ], dtype=np.uint8)
    # a:b value = [ [-13:3, -13:30], [30:30, 30:-19]
    expected_indices = np.array([[0, 1], [0, 0]])
    lab2 = GreenFromLab2({'short_name': 'L*a*b* v2 test', 'a_threshold_low': -9999, 'a_threshold_high': -11,
                          'b_threshold_low': 5, 'b_threshold_high': 57})
    actual_indices = lab2.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_assumes_green_func():
    # lab1 = L, a, b from R,G,B
    # 1 if -9999 <= a <= -6 and 5 <= b <= 57
    test_image = np.array([  # B G R
        [[0, 0, 0], [255, 0, 0]],
        [[0, 255, 0], [0, 0, 255]],
        [[128, 255, 128], [255, 255, 255]],
    ], dtype=np.uint8)
    expected_indices = np.array([[1, 1], [1, 1], [1, 1]])
    assumes_green = AssumesGreen({'short_name': 'Assumes green test'})
    actual_indices = assumes_green.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_matts_hue_mapping_func():
    # hue = RGB+I => IGB, then read IGB as RGB, create HSV, return 1 if HSV within range
    test_image = np.array([  # B G R Ir
        [[0, 0, 0, 255], [0, 0, 255, 0]],
        [[255, 0, 0, 0], [0, 255, 0, 0]]
    ], dtype=np.uint8)
    # HSV value = [ [[0,255,255], [0,0,0]], [[120,255,255], [60,255,255] ]
    expected_indices = np.array([[1, 0], [0, 0]])
    matt_index = MattIrHSV({'short_name': 'matt test',
                            'h_threshold_low': 0, 'h_threshold_high': 10,
                            's_threshold_low': 125, 's_threshold_high': 255,
                            'v_threshold_low': 20, 'v_threshold_high': 255})
    actual_indices = matt_index.index(test_image)
    np.testing.assert_allclose(actual_indices, expected_indices)


# def test_execution_speed_of_green_leaf_index_func():
#     previous_execution_times = {
#         'C02W60JUG8WL.local': 0.004,
#         'C02W60JUG8WL': 0.004,
#         'DESKTOP-QRHLKN8': 0.008
#     }
#
#     # platform_machine = platform.machine()
#     # platform_processor = platform.processor()
#     # platform_info = platform.platform()
#     platform_host_name = platform.node()
#
#     num_iterations = 30
#     elapsed_times = timeit.repeat(
#         stmt="apply_vegetation_index(gli, img, mask)",
#         setup="import numpy as np\n"
#               "from green_spaces.vegetation_analysis import GreenLeafIndex\n"
#               "from green_spaces.calculate_indices import apply_vegetation_index\n"
#               "img_size = (256, 256, 3)\n"
#               "img = np.zeros(img_size, dtype=np.uint8)\n"
#               "mask = np.full(img_size[:2], True, dtype=np.bool)\n"
#               "gli = GreenLeafIndex({'short_name': 'GLI speed test', 'threshold_low': 0.2, 'threshold_high': 0.8})\n",
#         number=num_iterations,
#         repeat=3
#     )
#
#     elapsed_time = min(elapsed_times)
#
#     time_per_iteration = elapsed_time / num_iterations
#
#     print(f'\nElapsed time per iteration: {time_per_iteration:0.4}s')
#
#     if platform_host_name not in previous_execution_times:
#         print(f'First execution on "{platform_host_name}"')
#     else:
#         assert time_per_iteration <= previous_execution_times[platform_host_name]
