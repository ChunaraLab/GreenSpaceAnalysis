"""
Collection of vegetation index calculation functions. Each function returns a value in the range 0..1
A collection of indices is presented at https://www.harrisgeospatial.com/docs/BroadbandGreenness.html
"""
import pickle
from os import path
from pathlib import PurePath

import cv2
import numpy as np
from keras.models import load_model


class GreenFromHSV(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= hue <= {self.__threshold_high}'

    def index(self, bgr_image):
        """
        :param bgr_image: 3D array of red values each in the range 0..255, of the image to be analysed, of the
            form [height, width, 2] ordered blue, green red
        :return: threshold flag for hue in green range, 1 for green, 0 for other
        """

        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        return np.where(np.logical_and(h > self.__threshold_low, h < self.__threshold_high), 1, 0)


class GreenLeafIndex(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']
        
    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= GLI <= {self.__threshold_high}'

    def index(self, bgr_image):
        """
        Definition taken from https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#Green6
        gli = ((green - red) + (green - blue)) / ((2.0 * green) + red + blue)

        Refactoring to produce common sub-expressions:
        gli = ((2.0 * _g) - (_r + _b)) / ((2.0 * _g) + (_r + _b))

        Analyses the given image, provided as 3 2D arrays, and returns an array of green leaf vegetation
        indices (one per pixel), a 2D array. Uses numpy ufunc to operate on the arrays directly for speed.

        :param bgr_image: 3D array of red values each in the range 0..255, of the image to be analysed, of the
            form [height, width, 2] ordered blue, green red
        :return: threshold flag for green leaf index in threshold range, 1 for vegetation, 0 for other
        """
        b = bgr_image[:, :, 0].astype(float)
        g = bgr_image[:, :, 1].astype(float)
        r = bgr_image[:, :, 2].astype(float)

        _r_plus_b = np.add(r, b)
        _g_times_2 = np.multiply(2.0, g)
        _g_times_2_subtract_r_plus_b = np.subtract(_g_times_2, _r_plus_b)
        _g_times_2_plus_r_plus_b = np.add(_g_times_2, _r_plus_b)
        del _r_plus_b
        del _g_times_2

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        gli = np.true_divide(_g_times_2_subtract_r_plus_b, _g_times_2_plus_r_plus_b)
        np.seterr(**old_settings)
        del _g_times_2_subtract_r_plus_b
        del _g_times_2_plus_r_plus_b

        return np.where(np.logical_and(gli > self.__threshold_low, gli < self.__threshold_high), 1, 0)


class NormalizedDifferenceVegetationIndexCIR(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']

    @staticmethod
    def input_format():
        return "CIR"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= NDVI <= {self.__threshold_high}'

    def index(self, irgr_image):
        """
        Definition taken from https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI
        ndvi = (nir - red) / (nir + red)

        The value of this index ranges from -1 to 1. The common range for green vegetation is 0.2 to 0.8.

        From https://docs.wixstatic.com/ugd/66c69f_e99f0d32a60f495c8c4334f6fc033d11.pdf
        Analyses the given image, provided as a 3D arrays (2D with channels g, r, ir), and returns an array of
        normalised difference vegetation indices (one per pixel), a 2D array. Uses numpy ufunc to operate on the arrays
        directly for speed. (ir, r, g) is CIR format, see https://www.altavian.com/knowledge-base/cir-imagery/

        Note that images are recorded in R,G,B format, but are loaded in order of B, G, R; with CIR, NIR is red channel,
        red is green channel, green is blue channel; hence retrieved in the order green, red, infra red

        :param irgr_image: 3D array of near green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 3] ordered near infrared, green, red
        :return: threshold flag for normalised difference vegetation index in threshold range, 1 for vegetation,
            0 for other
        """
        nir = irgr_image[:, :, 2].astype(float)
        r = irgr_image[:, :, 1].astype(float)
        # g = irgr_image[:, :, 0]

        _nir_minus_r = np.subtract(nir, r)
        _nir_plus_r = np.add(nir, r)

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        ndvi = np.true_divide(_nir_minus_r, _nir_plus_r)
        np.seterr(**old_settings)
        del _nir_minus_r
        del _nir_plus_r

        return np.where(np.logical_and(ndvi > self.__threshold_low, ndvi < self.__threshold_high), 1, 0)


class NormalizedDifferenceVegetationIndexIRGB(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']

    @staticmethod
    def input_format():
        return "IRGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= NDVI <= {self.__threshold_high}'

    def index(self, bgri_image):
        """
        Definition taken from https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI
        ndvi = (nir - red) / (nir + red)

        The value of this index ranges from -1 to 1. The common range for green vegetation is 0.2 to 0.8.

        From https://docs.wixstatic.com/ugd/66c69f_e99f0d32a60f495c8c4334f6fc033d11.pdf
        Analyses the given image, provided as a 3D arrays (2D with channels g, r, ir), and returns an array of
        normalised difference vegetation indices (one per pixel), a 2D array. Uses numpy ufunc to operate on the arrays
        directly for speed.

        Note that images are recorded in I,R,G,B format, but are loaded in order of B, G, R, I

        :param bgri_image: 3D array of blue, green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 3] ordered near blue, green, red, infrared
        :return: threshold flag for normalised difference vegetation index in threshold range, 1 for vegetation,
            0 for other
        """
        nir = bgri_image[:, :, 3].astype(float)
        r = bgri_image[:, :, 2].astype(float)

        _nir_minus_r = np.subtract(nir, r)
        _nir_plus_r = np.add(nir, r)

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        ndvi = np.true_divide(_nir_minus_r, _nir_plus_r)
        np.seterr(**old_settings)
        del _nir_minus_r
        del _nir_plus_r

        return np.where(np.logical_and(ndvi > self.__threshold_low, ndvi < self.__threshold_high), 1, 0)


class VisualNormalizedDifferenceVegetationIndex(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']
        # self.__threshold_low = 0.2
        # self.__threshold_high = 1

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= vNDVI <= {self.__threshold_high}'

    def index(self, bgr_image):
        """
        Definition taken from: https://support.precisionmapper.com/support/solutions/articles/6000214541-visual-ndvi
        and: https://support.precisionmapper.com/support/solutions/articles/
        6000187226-vegetation-indices-for-visual-data

        The Visual NDVI algorithm is based from a vegetation index called NGRDI-
        Normalized Green Red Difference Index. In this algorithm the Red and Green bands
        of a visual image are utilized to calculate a Vegetation Index value. This
        vegetation index value is designed to detect differences in green canopy area.
        It leans heavily on the green color of a healthy plant.
        The algorithm has been a tested indicator of chlorophyll content in several
        different crop types including Corn, Alfalfa, Soybean, and Wheat.

        vndvi = (green - red) / (green + red)

        Note that images are records in R,G,B format, but are loaded in order of B, G, R

        :param bgr_image: 3D array of near green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 3] ordered near infrared, green, red
        :return: threshold flag for visual normalised difference vegetation index in threshold range, 1 for vegetation,
            0 for other
        """
        # b = bgr_image[:, :, 0].astype(float)
        g = bgr_image[:, :, 1].astype(float)
        r = bgr_image[:, :, 2].astype(float)

        _g_minus_r = np.subtract(g, r)
        _g_plus_r = np.add(g, r)

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        vndvi = np.true_divide(_g_minus_r, _g_plus_r)
        np.seterr(**old_settings)
        del _g_minus_r
        del _g_plus_r

        return np.where(np.logical_and(vndvi > self.__threshold_low, vndvi < self.__threshold_high), 1, 0)


class VisualAtmosphericResistanceIndex(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__threshold_low = config['threshold_low']
        self.__threshold_high = config['threshold_high']

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__threshold_low} <= VARI <= {self.__threshold_high}'

    def index(self, bgr_image):
        """
        Definition taken from: https://support.precisionmapper.com/support/solutions/articles/6000214543-vari
        and: https://support.precisionmapper.com/support/solutions/articles/
        6000187226-vegetation-indices-for-visual-data

        The Visual Atmospheric Resistance Index is a vegetative index that was originally
        designed for satellite imagery. It is found to be minimally sensitive to atmospheric
        effects, allowing the estimation of vegetation fraction in a wide range of
        atmospheric conditions.

        vari = (green - red) / (green + red - blue)

        As sunlight reaches the earth’s atmosphere it is scattered in all directions by
        the gasses and particles in the air. But blue light tends to scatter more than
        all the other colors because it travels in smaller wavelengths than the rest of
        the visual spectrum. Therefore, we see the sky as blue most of the time. By
        including the Blue band in the denominator of the VARI equation we are accounting
        for the effects of the atmosphere on this Vegetation Index calculation.
        For each pixel VARI is calculating a ratio of green vegetation cover. Values can
        be anywhere from -1.0 and 1.0 in the final output.

        Note that images are records in R,G,B format, but are loaded in order of B, G, R

        :param bgr_image: 3D array of near green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 3] ordered near infrared, green, red
        :return: threshold flag for Visual Atmospheric Resistance Index in threshold range, 1 for vegetation,
            0 for other
        """
        b = bgr_image[:, :, 0].astype(float)
        g = bgr_image[:, :, 1].astype(float)
        r = bgr_image[:, :, 2].astype(float)

        _g_minus_r = np.subtract(g, r)
        _g_plus_r = np.add(g, r)
        _g_plus_r_minus_b = np.subtract(_g_plus_r, b)

        old_settings = np.seterr(divide='ignore', invalid='ignore')
        vari = np.true_divide(_g_minus_r, _g_plus_r_minus_b)
        np.seterr(**old_settings)
        del _g_minus_r
        del _g_plus_r
        del _g_plus_r_minus_b

        return np.where(np.logical_and(vari > self.__threshold_low, vari < self.__threshold_high), 1, 0)


class GreenFromLab1(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__a_threshold_low = config['a_threshold_low']
        self.__a_threshold_high = config['a_threshold_high']

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__a_threshold_low} <= a <= {self.__a_threshold_high}'

    def index(self, bgr_image):
        """
        The image is first converted to the L*a*b* colour space. Lab is better suited
        to image processing tasks since it is much more intuitive than RGB. In Lab,
        the lightness of a pixel (L value) is seperated from the colour (A and B values).
        A negative A value represents degrees of green, positive A, degrees of red.
        Negative B represents blue, while positive B represents yellow. A colour can
        never be red _and_ green or yellow _and_ blue at the same time. Therefore the
        Lab colour space provides a more intuitive separability than RGB (where all
        values must be adjusted to encode a colour.) Furthermore, since lightness value
        (L) is represented independently from colour, a 'green' value will be robust to
        varying lighting conditions.

        :param bgr_image: 3D array of red values each in the range 0..255, of the image to be analysed, of the
            form [height, width, 2] ordered blue, green red
        :return: threshold flag for L*a*b in threshold range, 1 for vegetation, 0 for other
        """

        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)

        a = a - 128.0

        return np.where(np.logical_and(a > self.__a_threshold_low, a < self.__a_threshold_high), 1, 0)


class GreenFromLab2(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__a_threshold_low = config['a_threshold_low']
        self.__a_threshold_high = config['a_threshold_high']
        self.__b_threshold_low = config['b_threshold_low']
        self.__b_threshold_high = config['b_threshold_high']

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__a_threshold_low} <= a <= {self.__a_threshold_high}' \
            f' and {self.__b_threshold_low} <= b <= {self.__b_threshold_high}'

    def index(self, bgr_image):
        """
        The image is first converted to the L*a*b* colour space. Lab is better suited
        to image processing tasks since it is much more intuitive than RGB. In Lab,
        the lightness of a pixel (L value) is seperated from the colour (A and B values).
        A negative A value represents degrees of green, positive A, degrees of red.
        Negative B represents blue, while positive B represents yellow. A colour can
        never be red _and_ green or yellow _and_ blue at the same time. Therefore the
        Lab colour space provides a more intuitive separability than RGB (where all
        values must be adjusted to encode a colour.) Furthermore, since lightness value
        (L) is represented independently from colour, a 'green' value will be robust to
        varying lighting conditions.

        :param bgr_image: 3D array of red values each in the range 0..255, of the image to be analysed, of the
            form [height, width, 2] ordered blue, green red
        :return: threshold flag for L*a*b in threshold range, 1 for vegetation, 0 for other
        """

        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)

        a = a - 128.0
        b = b - 128.0

        return np.where(
            np.logical_and(
                np.logical_and(a > self.__a_threshold_low, a < self.__a_threshold_high),
                np.logical_and(b > self.__b_threshold_low, b < self.__b_threshold_high)
            ),
            1, 0)


class AssumesGreen(object):
    def __init__(self, config):
        self.__short_name = config['short_name']

    @staticmethod
    def input_format():
        return "RGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'all pixels assumed vegetation'

    @staticmethod
    def index(bgr_image):
        """
        Assumes all pixels are green, compares to the naive assumption that all pixels within a garden polygon are grass

        :param bgr_image: 3D array of red values each in the range 0..255, of the image to be analysed, of the
            form [height, width, 2] ordered blue, green red
        :return: array of 1s indicating 100% vegetation cover
        """

        return np.ones(bgr_image.shape[0:2])


class MattIrHSV(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__h_threshold_low = config['h_threshold_low']
        self.__h_threshold_high = config['h_threshold_high']
        self.__s_threshold_low = config['s_threshold_low']
        self.__s_threshold_high = config['s_threshold_high']
        self.__v_threshold_low = config['v_threshold_low']
        self.__v_threshold_high = config['v_threshold_high']

    @staticmethod
    def input_format():
        return "IRGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if {self.__h_threshold_low} <= H <= {self.__h_threshold_high}' \
            f' and {self.__s_threshold_low} <= S <= {self.__s_threshold_high}' \
            f' and {self.__v_threshold_low} <= V <= {self.__v_threshold_high}'

    def index(self, bgri_image):
        """
        c/o Matt:
        1. Rearrange bands from RGB+I to I G B (in that order, discarding red).
        2. Convert this rearranged banded image to HSV colour space    
            (I used:  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        3. Define and append the lower and upper values of ‘red’ to variables (as we’re casting the IR signal in to 
            the red band, so then on thresholding based on intensity of red). 
            I used: (RED_MIN = np.array([0, 125, 20],np.uint8) <and> RED_MAX = np.array([10, 255, 255],np.uint8). 
            I think the value of RED_MAX could be tweaked to capture really dark red areas (shadows) where an IR signal 
            still exists beneath (albeit weak).
        4. I then did the thresholding using
            frame_threshed = cv2.inRange(hsv_img, RED_MIN, RED_MAX)
        5. With the binary layer that this produces I derived % of each class (urban vs. vegetation) using pixel count 
            against total pixel count per garden.

        Note that images are records in R,G,B,Ir format, but are loaded in order of B, G, R, Ir

        :param bgri_image: 3D array of blue, green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 4] ordered near blue, green, red, infrared
        :return: vegetation index of 0 or 1, indicating vegetation true or false
        """
        channel_shape = (bgri_image.shape[0], bgri_image.shape[1], 1)
        nir = bgri_image[:, :, 3].reshape(channel_shape)
        g = bgri_image[:, :, 1].reshape(channel_shape)
        b = bgri_image[:, :, 0].reshape(channel_shape)

        image_bgi = np.concatenate((b, g, nir), axis=2)

        hsv_img = cv2.cvtColor(image_bgi, cv2.COLOR_BGR2HSV)

        red_min = (self.__h_threshold_low, self.__s_threshold_low, self.__v_threshold_low)
        red_max = (self.__h_threshold_high, self.__s_threshold_high, self.__v_threshold_high)

        image_in_threshold = cv2.inRange(hsv_img, red_min, red_max)

        return np.where(image_in_threshold > 0, 1, 0)


class Matt2IrHSV(object):
    def __init__(self, config):
        self.__short_name = config['short_name']
        self.__h1_threshold_low = config['h1_threshold_low']
        self.__h1_threshold_high = config['h1_threshold_high']
        self.__s1_threshold_low = config['s1_threshold_low']
        self.__s1_threshold_high = config['s1_threshold_high']
        self.__v1_threshold_low = config['v1_threshold_low']
        self.__v1_threshold_high = config['v1_threshold_high']
        self.__h2_threshold_low = config['h2_threshold_low']
        self.__h2_threshold_high = config['h2_threshold_high']
        self.__s2_threshold_low = config['s2_threshold_low']
        self.__s2_threshold_high = config['s2_threshold_high']
        self.__v2_threshold_low = config['v2_threshold_low']
        self.__v2_threshold_high = config['v2_threshold_high']

    @staticmethod
    def input_format():
        return "CIR"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f'vegetation if ({self.__h1_threshold_low} <= H <= {self.__h1_threshold_high}' \
            f' and {self.__s1_threshold_low} <= S <= {self.__s1_threshold_high}' \
            f' and {self.__v1_threshold_low} <= V <= {self.__v1_threshold_high})' \
            f' or ({self.__h2_threshold_low} <= H <= {self.__h2_threshold_high}' \
            f' and {self.__s2_threshold_low} <= S <= {self.__s2_threshold_high}' \
            f' and {self.__v2_threshold_low} <= V <= {self.__v2_threshold_high})'

    def index(self, irgr_image):
        """
        A mapping from CIR directly interpreted as RGB and then to HSV.
        Two HSV ranges are used to cope with the discontinuity around 360 degrees
        (stored as 0-180 to fit in the range 0-255) in the red hue.

        Note that images are records in Ir,R,G format, but are loaded in order of G, R, Ir.

        :param irgr_image: 3D array of green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 4].
        :return: vegetation index of 0 or 1, indicating vegetation true or false.
        """
        hsv = cv2.cvtColor(irgr_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv,
                            (self.__h1_threshold_low, self.__s1_threshold_low, self.__v1_threshold_low),
                            (self.__h1_threshold_high, self.__s1_threshold_high, self.__v1_threshold_high)
                            )

        # far end hsv red net
        mask2 = cv2.inRange(hsv,
                            (self.__h2_threshold_low, self.__s2_threshold_low, self.__v2_threshold_low),
                            (self.__h2_threshold_high, self.__s2_threshold_high, self.__v2_threshold_high)
                            )

        # final mask and masked
        mask = cv2.bitwise_or(mask1, mask2)
        return np.where(mask > 0, 1, 0)


class NeuralNetworkClassifier(object):

    @staticmethod
    def load_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    def __init__(self, config, load_pickle_fn=load_pickle, load_model_fn=load_model):
        self.load_pickle_fn = load_pickle_fn

        self.__short_name = config['short_name']

        config_path = PurePath(path.realpath(__file__)).parent / 'config'
        self.__nn_state_path = config_path / config['nn_state_path']

        self.__nn_model = load_model_fn(str(self.__nn_state_path))

        self.__monochrome_pca_components_path = config_path / config['monochrome_pca_components_path']
        self.__monochrome_pca_components = self.load_pickle(self.__monochrome_pca_components_path).T
        self.__monochrome_pca_mean_path = config_path / config['monochrome_pca_mean_path']
        self.__monochrome_pca_mean = self.load_pickle(self.__monochrome_pca_mean_path).T
        self.__monochrome_pca_min = config['monochrome_pca_min']
        self.__monochrome_pca_max = config['monochrome_pca_max']

        self.__brightness_pca_components_path = config_path / config['brightness_pca_components_path']
        self.__brightness_pca_components = self.load_pickle(self.__brightness_pca_components_path).T
        self.__brightness_pca_mean_path = config_path / config['brightness_pca_mean_path']
        self.__brightness_pca_mean = self.load_pickle(self.__brightness_pca_mean_path).T
        self.__brightness_pca_inputs_min = config['brightness_pca_inputs_min']
        self.__brightness_pca_inputs_max = config['brightness_pca_inputs_max']
        self.__brightness_pca_min = config['brightness_pca_min']
        self.__brightness_pca_max = config['brightness_pca_max']

        self.__colour_pca_components_path = config_path / config['colour_pca_components_path']
        self.__colour_pca_components = self.load_pickle(self.__colour_pca_components_path).T
        self.__colour_pca_mean_path = config_path / config['colour_pca_mean_path']
        self.__colour_pca_mean = self.load_pickle(self.__colour_pca_mean_path).T
        self.__colour_pca_inputs_min = config['colour_pca_inputs_min']
        self.__colour_pca_inputs_max = config['colour_pca_inputs_max']
        self.__colour_pca_min = config['colour_pca_min']
        self.__colour_pca_max = config['colour_pca_max']

    @staticmethod
    def input_format():
        return "IRGB"

    @property
    def short_name(self):
        return self.__short_name

    @property
    def configuration(self):
        return f"Neural network model='{self.__nn_state_path}'," \
            f" brightness PCA='{self.__brightness_pca_components_path}," \
            f" monochrome PCA='{self.__monochrome_pca_components_path}," \
            f" colour PCA='{self.__colour_pca_components_path}'"

    @staticmethod
    def concatenate_channels(list_of_channels):
        new_shape = list_of_channels[0].shape + (1,)
        return np.concatenate([x.reshape(new_shape) for x in list_of_channels], axis=2)

    @staticmethod
    def generate_8bit_pca_from_n_normalised_channels(concatenated_channels, pca_components, pca_mean,
                                                     pca_components_min, pca_components_max, inputs_min, inputs_max):
        scaled_channels = np.zeros(concatenated_channels.shape)
        for channel in range(concatenated_channels.shape[2]):
            scaled_inputs = (concatenated_channels[:, :, channel].astype(np.float) - inputs_min[channel]) \
                            / (inputs_max[channel] - inputs_min[channel])
            scaled_channels[:, :, channel] = scaled_inputs

        return NeuralNetworkClassifier.generate_8bit_pca_from_n_channels(scaled_channels, pca_components, pca_mean,
                                                                         pca_components_min, pca_components_max)

    @staticmethod
    def generate_8bit_pca_from_n_channels(concatenated_channels, pca_components, pca_mean,
                                          pca_components_min, pca_components_max):
        original_shape = concatenated_channels.shape
        number_of_rows = original_shape[0] * original_shape[1]
        pixel_list = concatenated_channels.reshape((number_of_rows, original_shape[2]))

        # Perform PCA

        # subtract mean first
        pixels_post_mean = np.subtract(pixel_list, pca_mean)

        # then dot product
        pca_results = np.matmul(pixels_post_mean, pca_components)

        # Scale to range
        output_shape = (number_of_rows, pca_components.shape[1] - 1)
        scaled_pca_results = np.zeros(shape=output_shape)
        for n, (pca_max, pca_min) in enumerate(list(zip(pca_components_max, pca_components_min))[1:]):
            t = 255.0 * ((pca_results[:, n + 1] - pca_min) / (pca_max - pca_min))
            scaled_pca_results[:, n] = t

        array_of_pca_results = scaled_pca_results.reshape(
            (original_shape[0], original_shape[1], pca_components.shape[1] - 1))

        return np.rint(array_of_pca_results).astype(np.uint8)

    def index(self, bgri_image):
        """

        :param bgri_image: 3D array of blue, green, red, infra red values each in the range 0..255, of the image to be
            analysed, of the form [height, width, 4] ordered near blue, green, red, infrared
        :return: vegetation index of 0 or 1, indicating vegetation true or false
        """

        bgr_image = bgri_image[:, :, 0:3]
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

        monochrome_pca_results = self.generate_8bit_pca_from_n_channels(
            self.concatenate_channels([bgri_image[:, :, 2], bgri_image[:, :, 1], bgri_image[:, :, 0]]),
            self.__monochrome_pca_components, self.__monochrome_pca_mean,
            self.__monochrome_pca_min, self.__monochrome_pca_max)

        brightness_pca_results = self.generate_8bit_pca_from_n_normalised_channels(
            self.concatenate_channels([bgri_image[:, :, 2], bgri_image[:, :, 1], bgri_image[:, :, 0],
                                       hsv_image[:, :, 2], lab_image[:, :, 0]]),
            self.__brightness_pca_components, self.__brightness_pca_mean,
            self.__brightness_pca_min, self.__brightness_pca_max, self.__brightness_pca_inputs_min,
            self.__brightness_pca_inputs_max)

        colour_pca_results = self.generate_8bit_pca_from_n_normalised_channels(
            self.concatenate_channels([bgri_image[:, :, 2], bgri_image[:, :, 1], bgri_image[:, :, 0],
                                       hsv_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]]),
            self.__colour_pca_components, self.__colour_pca_mean,
            self.__colour_pca_min, self.__colour_pca_max,
            self.__colour_pca_inputs_min, self.__colour_pca_inputs_max
        )

        input_channels = self.concatenate_channels(
            (
                bgri_image[:, :, 2], bgri_image[:, :, 1], bgri_image[:, :, 0],  # R, G, B
                monochrome_pca_results, bgri_image[:, :, 3],  # mono PCA, ir
            ),
        )

        input_channels = np.concatenate((input_channels, brightness_pca_results, colour_pca_results), axis=2)

        number_of_pixels = bgri_image.shape[0] * bgri_image.shape[1]
        reshaped_channels = input_channels.reshape((number_of_pixels, 11))
        prediction = self.__nn_model.predict_classes(reshaped_channels)

        vegetation_index = np.where(prediction < 2, 1, 0)

        reshaped_vegetation_index = vegetation_index.reshape(
            (bgri_image.shape[0], bgri_image.shape[1])
        )
        return reshaped_vegetation_index
