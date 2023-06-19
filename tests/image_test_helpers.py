import numpy as np


image_left_half_red_right_half_blue_4x4 = np.array([
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
], dtype=np.uint8)

image_top_left_quarter_red_remainder_blue_4x4 = np.array([
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
    [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
], dtype=np.uint8)

mask_top_half_true_4x4 = np.array([
    [True, True, True, True],
    [True, True, True, True],
    [False, False, False, False],
    [False, False, False, False],
], dtype=np.bool)

mask_all_false_4x4 = np.array([
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
], dtype=np.bool)

mask_all_true_4x4 = np.array([
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
], dtype=np.bool)


class CountRedPixels(object):
    @staticmethod
    def index(bgr_image):
        b = bgr_image[:, :, 0]
        g = bgr_image[:, :, 1]
        r = bgr_image[:, :, 2]

        red_not_green_pixel = np.logical_and(np.equal(r, 255), np.equal(g, 0))
        red_pixel = np.logical_and(np.equal(b, 0), red_not_green_pixel)

        ones = np.ones(r.shape)
        zeros = np.zeros(r.shape)
        red_counts = np.where(red_pixel, ones, zeros)

        return red_counts
