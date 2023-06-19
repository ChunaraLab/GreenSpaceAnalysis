import os
from pathlib import PurePath

import cv2
import numpy as np
import pandas as pd
import pytest
from keras.models import load_model

from green_spaces.vegetation_analysis import NeuralNetworkClassifier

# monochrome_pca_components = nnc.load_pickle(config_path / "pca_mono.pkl").T
monochrome_pca_components = np.array(
    [[0.62646133, -0.3017635],
     [0.583181, -0.43027943],
     [0.51715194, 0.85076342]]
)
monochrome_pca_mean = np.array(
    [105.66660477, 109.28897499, 92.92607798]
)
monochrome_pca_min = [-151.6851, -43.19246]
monochrome_pca_max = [262.3443, 69.40648]

# brightness_pca_components = nnc.load_pickle(config_path / "pca_bright.pkl").T
brightness_pca_components = np.array(
    [[0.46864159, -0.08728196, 0.83335901, -0.22887342],
     [0.45136526, -0.30787501, -0.48627001, -0.29764206],
     [0.38328559, 0.90189816, -0.15342144, -0.12641922],
     [0.46854311, -0.08140501, -0.0333985, 0.87893031],
     [0.45848006, -0.278474, -0.2107146, -0.26556703]]
)
brightness_pca_mean = np.array(
    [0.39541136, 0.39287073, 0.34383028, 0.38978753, 0.42404937]
)
brightness_pca_inputs_min = [8, 15, 8, 22, 11]
brightness_pca_inputs_max = [255, 255, 255, 255, 255]
brightness_pca_min = [-0.8459840, -0.1820107, -0.2020519, -0.02920802]
brightness_pca_max = [1.358846, 0.2513977, 0.3559869, 0.2736014]

# colour_pca_components = nnc.load_pickle(config_path / "pca_colour.pkl").T
colour_pca_components = np.array(
    [[0.56442686, 0.20712503, 0.00254163, 0.35315174],
     [0.54110403, 0.17733909, -0.24518915, -0.16812712],
     [0.4311484, 0.39495338, 0.09297395, -0.27337985],
     [-0.3983794, 0.70567344, -0.54331412, 0.21924551],
     [-0.00942986, 0.20955154, 0.55500983, 0.69349598],
     [0.20962997, -0.47725784, -0.57271766, 0.49322948]])
colour_pca_mean = np.array(
    [0.39541136, 0.39287073, 0.34383028, 0.27379716, 0.26986141, 0.51592484]
)
colour_pca_inputs_min = [8, 15, 8, 0, 105, 103]
colour_pca_inputs_max = [255, 255, 255, 179, 174, 168]
colour_pca_min = [-0.7957638, -0.4348105, -0.3928765, -0.3462591]
colour_pca_max = [1.033623, 1.051972, 0.4325098, 0.8134353]


def load_pca_pickle(pickle_path):
    if pickle_path.endswith('pca_mono.pkl'):
        return monochrome_pca_components
    elif pickle_path.endswith('pca_bright.pkl'):
        return brightness_pca_components
    elif pickle_path.endswith('pca_colour.pkl'):
        return colour_pca_components
    if pickle_path.endswith('pca_mono_mean.pkl'):
        return monochrome_pca_mean
    elif pickle_path.endswith('pca_bright_mean.pkl'):
        return brightness_pca_mean
    elif pickle_path.endswith('pca_colour_mean.pkl'):
        return colour_pca_mean
    else:
        raise ValueError(f'Unknown path to mock: "{pickle_path}"')


#  R,G,B,pca_mono,Ir,pca_bright_1,pca_bright_2,pca_bright_3,pca_colour_1,pca_colour_2,pca_colour_3
#         ID,labelB,labelG,labelR,labelNone

# 68864 [1] != [0] (B,G,R,Ir = 85, 99, 88, 96)
#  68866: 88,99,85, 105, 96, 111,82,18, 72,117,67
#         osgb4000000272540461,0,0,0,1   => None labelled but considered 'vegetation'

# 68879 [0] != [1] (B,G,R,Ir = 44, 53, 57, 90)
#  68881: 57,53,44, 92, 90, 105,110,18, 32,172,79
#         osgb4000000272540461,1,0,0,0   => Blue labelled but considered 'not vegetation'

# line no.: R,G,B,pca_mono,Ir,pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72,156,107,30,199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94,77,105,22,53,116,90; osgb4000000273670894,0,1,0,0
@pytest.mark.parametrize("red, green, blue, pca_mono", [
    ([[155]], [[140]], [[147]], [[138]]),
    ([[124]], [[121]], [[90]], [[68]]),
    ([[155, 124]], [[140, 121]], [[147, 90]], [[138, 68]]),
    ([[88]], [[99]], [[85]], [105]),
    ([[57]], [[53]], [[44]], [92]),
])
def test_neural_network_creates_mono_pca(red, green, blue, pca_mono):
    red_array = np.array(red).astype(np.uint8)
    green_array = np.array(green).astype(np.uint8)
    blue_array = np.array(blue).astype(np.uint8)
    expected_pca_array = np.array(pca_mono).astype(np.uint8).reshape(red_array.shape + (1,))

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_channels(
        NeuralNetworkClassifier.concatenate_channels([red_array, green_array, blue_array]),
        monochrome_pca_components, monochrome_pca_mean,
        monochrome_pca_min, monochrome_pca_max)

    assert np.array_equal(scaled_pca, expected_pca_array)


# line no.: R,G,B,pca_mono,Ir, pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72, 156,107,30, 199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94, 77,105,22, 53,116,90; osgb4000000273670894,0,1,0,0
@pytest.mark.parametrize("red, green, blue, pca_bright", [
    ([[155]], [[140]], [[147]], [[[156, 107, 30]]]),
    ([[124]], [[121]], [[90]], [[[77, 105, 22]]]),
    ([[155, 124]], [[140, 121]], [[147, 90]], [[[156, 107, 30], [77, 105, 22]]]),
    ([[88]], [[99]], [[85]], [[[111, 82, 18]]]),
    ([[57]], [[53]], [[44]], [[[105, 110, 18]]]),
])
def test_neural_network_creates_brightness_pca(red, green, blue, pca_bright):
    red_array = np.array(red).astype(np.uint8)
    green_array = np.array(green).astype(np.uint8)
    blue_array = np.array(blue).astype(np.uint8)
    expected_pca_array = np.array(pca_bright).astype(np.uint8)

    bgr_image = NeuralNetworkClassifier.concatenate_channels([blue_array, green_array, red_array])
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    value_array = hsv_image[:, :, 2]
    lightness_array = lab_image[:, :, 0]

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_normalised_channels(
        NeuralNetworkClassifier.concatenate_channels(
            [red_array, green_array, blue_array, value_array, lightness_array]),
        brightness_pca_components, brightness_pca_mean,
        brightness_pca_min, brightness_pca_max, brightness_pca_inputs_min, brightness_pca_inputs_max)

    # Spreadsheet doesn't match so something up let alone the code :(
    # Code matches spreadsheet (within 1 unit)
    assert np.array_equal(scaled_pca, expected_pca_array)


# line no.: R,G,B,pca_mono,Ir, pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72, 156,107,30, 199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94, 77,105,22, 53,116,90; osgb4000000273670894,0,1,0,0
@pytest.mark.parametrize("red, green, blue, pca_colour", [
    ([[155]], [[140]], [[147]], [[[199, 65, 113]]]),
    ([[124]], [[121]], [[90]], [[[53, 116, 90]]]),
    ([[155, 124]], [[140, 121]], [[147, 90]], [[[199, 65, 113], [53, 116, 90]]]),
    ([[88]], [[99]], [[85]], [[[72, 117, 67]]]),
    ([[57]], [[53]], [[44]], [[[32, 172, 79]]]),
    ([[90]], [[128]], [[77]], [[[43, 20, 55]]]),  # 76413
])
def test_neural_network_creates_colour_pca(red, green, blue, pca_colour):
    red_array = np.array(red).astype(np.uint8)
    green_array = np.array(green).astype(np.uint8)
    blue_array = np.array(blue).astype(np.uint8)
    expected_pca_array = np.array(pca_colour).astype(np.uint8)

    bgr_image = NeuralNetworkClassifier.concatenate_channels([blue_array, green_array, red_array])
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    hue_array = hsv_image[:, :, 0]
    a_array = lab_image[:, :, 1]
    b_array = lab_image[:, :, 2]

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_normalised_channels(
        NeuralNetworkClassifier.concatenate_channels([red_array, green_array, blue_array, hue_array, a_array, b_array]),
        colour_pca_components, colour_pca_mean, colour_pca_min, colour_pca_max, colour_pca_inputs_min,
        colour_pca_inputs_max)

    # Spreadsheet doesn't match so something up let alone the code :(
    assert np.array_equal(scaled_pca, expected_pca_array)


# line no.: R,G,B,pca_mono,Ir, pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72, 156,107,30, 199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94, 77,105,22, 53,116,90; osgb4000000273670894,0,1,0,0
@pytest.mark.parametrize(
    "red, green, blue, mono_pca, infra_red, bright_pca_1, bright_pca_2, bright_pca_3, "
    "colour_pca_1, colour_pca_2, colour_pca_3",
    [
        (155, 140, 147, 138, 72, 156, 107, 30, 199, 65, 113),
        (124, 121, 90, 68, 94, 77, 105, 22, 53, 116, 90),
        (88, 99, 85, 105, 96, 111, 82, 18, 72, 117, 67),
        (57, 53, 44, 92, 90, 105, 110, 18, 32, 172, 79),
    ])
def test_neural_network_pca_integration(red, green, blue, mono_pca, infra_red, bright_pca_1, bright_pca_2, bright_pca_3,
                                        colour_pca_1, colour_pca_2, colour_pca_3):
    class NeuralNetworkStub(object):
        def __init__(self):
            self.input_channels = None

        def predict_classes(self, input_channels):
            self.input_channels = input_channels
            return np.zeros(shape=[input_channels.shape[0], 1])

    nn_stub_instance = NeuralNetworkStub()

    def load_nn_model(_):
        return nn_stub_instance

    nnc = NeuralNetworkClassifier({"short_name": "nn",
                                   "full_name": "Neural network vegetation classifier",
                                   "nn_state_path": "12_8_4_all_ANN.h5",

                                   "monochrome_pca_components_path": "pca_mono.pkl",
                                   "monochrome_pca_mean_path": "pca_mono_mean.pkl",
                                   "monochrome_pca_min": monochrome_pca_min,
                                   "monochrome_pca_max": monochrome_pca_max,

                                   "brightness_pca_components_path": "pca_bright.pkl",
                                   "brightness_pca_mean_path": "pca_bright_mean.pkl",
                                   "brightness_pca_inputs_min": brightness_pca_inputs_min,
                                   "brightness_pca_inputs_max": brightness_pca_inputs_max,
                                   "brightness_pca_min": brightness_pca_min,
                                   "brightness_pca_max": brightness_pca_max,

                                   "colour_pca_components_path": "pca_colour.pkl",
                                   "colour_pca_mean_path": "pca_colour_mean.pkl",
                                   "colour_pca_inputs_min": colour_pca_inputs_min,
                                   "colour_pca_inputs_max": colour_pca_inputs_max,
                                   "colour_pca_min": colour_pca_min,
                                   "colour_pca_max": colour_pca_max,
                                   },
                                  load_pickle_fn=load_pca_pickle,
                                  load_model_fn=load_nn_model)

    infra_red_array = np.array(infra_red).reshape((1, 1, 1)).astype(np.uint8)
    red_array = np.array(red).reshape((1, 1, 1)).astype(np.uint8)
    green_array = np.array(green).reshape((1, 1, 1)).astype(np.uint8)
    blue_array = np.array(blue).reshape((1, 1, 1)).astype(np.uint8)

    bgri_image = np.concatenate([blue_array, green_array, red_array, infra_red_array], axis=2)
    result = nnc.index(bgri_image)

    # check that the nn_stub_instance is called with correct PCA...
    expected_nn_parameters = np.array([[
        red, green, blue, mono_pca, infra_red, bright_pca_1, bright_pca_2, bright_pca_3,
        colour_pca_1, colour_pca_2, colour_pca_3
    ]])

    assert np.array_equal(nn_stub_instance.input_channels, expected_nn_parameters)


# line no.: R,G,B,pca_mono,Ir, pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72, 156,107,30, 199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94, 77,105,22, 53,116,90; osgb4000000273670894,0,1,0,0

# 68879 [0] != [1] (B,G,R,Ir = 44, 53, 57, 90)
# 68880 [0] != [1] (B,G,R,Ir = 44, 53, 57, 86)
# 68881 [0] != [1] (B,G,R,Ir = 60, 69, 72, 70)
# 68882 [0] != [1] (B,G,R,Ir = 81, 89, 89, 50)
# 68883 [0] != [1] (B,G,R,Ir = 96, 104, 103, 32)
# 68884 [0] != [1] (B,G,R,Ir = 100, 110, 104, 23)

# 68881: 57,53,44,92,90,105,110,18,32,172,79      2
# 68882: 57,53,44,92,86,105,110,18,32,172,79      2
# 68883: 72,69,60,97,70,109,105,14,43,170,76      2
# 68884: 89,89,81,106,50,118,98,5,59,160,71       2
# 68885: 103,104,96,111,32,122,94,5,71,156,69     3
# 68886: 104,110,100,112,23,122,87,12,80,133,68   3


@pytest.mark.parametrize(
    "red, green, blue, mono_pca, infra_red, bright_pca_1, bright_pca_2, bright_pca_3, "
    "colour_pca_1, colour_pca_2, colour_pca_3, expected_prediction",
    [
        (155, 140, 147, 138, 72, 156, 107, 30, 199, 65, 113, 3),
        (124, 121, 90, 68, 94, 77, 105, 22, 53, 116, 90, 1),

        (57, 53, 44, 92, 90, 105, 110, 18, 32, 172, 79, 2),
        (57, 53, 44, 92, 86, 105, 110, 18, 32, 172, 79, 2),
        (72, 69, 60, 97, 70, 109, 105, 14, 43, 170, 76, 2),
        (89, 89, 81, 106, 50, 118, 98, 5, 59, 160, 71, 2),
        (103, 104, 96, 111, 32, 122, 94, 5, 71, 156, 69, 3),
        (104, 110, 100, 112, 23, 122, 87, 12, 80, 133, 68, 3),
    ])
def test_neural_network_test_underlying_nn(red, green, blue, mono_pca, infra_red,
                                           bright_pca_1, bright_pca_2, bright_pca_3,
                                           colour_pca_1, colour_pca_2, colour_pca_3,
                                           expected_prediction):
    nn_state_path = PurePath(os.path.realpath(__file__)).parent.parent / 'green_spaces' / 'config' / '12_8_4_all_ANN.h5'

    nn = load_model(str(nn_state_path))

    inputs = np.array([[red, green, blue, mono_pca, infra_red,
                        bright_pca_1, bright_pca_2, bright_pca_3,
                        colour_pca_1, colour_pca_2, colour_pca_3]])

    predicted_class = nn.predict_classes(inputs)

    assert predicted_class == expected_prediction


# line no.: R,G,B,pca_mono,Ir, pca_bright_1,pca_bright_2,pca_bright_3,
#           pca_colour_1,pca_colour_2,pca_colour_3; ID,labelB,labelG,labelR,labelNone
# 2:      155,140,147,138,72, 156,107,30, 199,65,113; osgb4000000267695338,0,0,0,1
# 241944: 124,121,90,68,94, 77,105,22, 53,116,90; osgb4000000273670894,0,1,0,0
@pytest.mark.parametrize(
    "red, green, blue, infra_red, expected_vegetation_label",
    [
        ([[155]], [[140]], [[147]], [[72]], [[0]]),
        ([[124]], [[121]], [[90]], [[94]], [[1]]),
        ([[155, 124]], [[140, 121]], [[147, 90]], [[72, 94]], [[0, 1]]),
    ])
def test_neural_network_full_integration(red, green, blue, infra_red, expected_vegetation_label):
    nnc = NeuralNetworkClassifier({"short_name": "nn",
                                   "full_name": "Neural network vegetation classifier",
                                   "nn_state_path": "12_8_4_all_ANN.h5",

                                   "monochrome_pca_components_path": "pca_mono.pkl",
                                   "monochrome_pca_mean_path": "pca_mono_mean.pkl",
                                   "monochrome_pca_min": monochrome_pca_min,
                                   "monochrome_pca_max": monochrome_pca_max,

                                   "brightness_pca_components_path": "pca_bright.pkl",
                                   "brightness_pca_mean_path": "pca_bright_mean.pkl",
                                   "brightness_pca_inputs_min": brightness_pca_inputs_min,
                                   "brightness_pca_inputs_max": brightness_pca_inputs_max,
                                   "brightness_pca_min": brightness_pca_min,
                                   "brightness_pca_max": brightness_pca_max,

                                   "colour_pca_components_path": "pca_colour.pkl",
                                   "colour_pca_mean_path": "pca_colour_mean.pkl",
                                   "colour_pca_inputs_min": colour_pca_inputs_min,
                                   "colour_pca_inputs_max": colour_pca_inputs_max,
                                   "colour_pca_min": colour_pca_min,
                                   "colour_pca_max": colour_pca_max,
                                   },
                                  load_pickle_fn=load_pca_pickle)

    red_array = np.array(red).astype(np.uint8)
    green_array = np.array(green).astype(np.uint8)
    blue_array = np.array(blue).astype(np.uint8)
    infra_red_array = np.array(infra_red).astype(np.uint8)
    expected_vegetation_label_array = np.array(expected_vegetation_label).astype(np.uint8)

    bgri_image = NeuralNetworkClassifier.concatenate_channels((blue_array, green_array, red_array, infra_red_array))
    vegetation_label_array = nnc.index(bgri_image)

    assert np.array_equal(vegetation_label_array, expected_vegetation_label_array)


def test_neural_network_creates_mono_pca_full_training_set():
    features_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_inputs.csv'

    feature_df = pd.read_csv(str(features_path), header=0, delimiter=',')
    feature_df.drop(columns=['Ir', 'pca_bright_1', 'pca_bright_2', 'pca_bright_3', 'pca_colour_1', 'pca_colour_2',
                             'pca_colour_3'], inplace=True)  # leaves: R,G,B,pca_mono
    column_order = ['R', 'G', 'B', 'pca_mono']
    rgb_pca_df = feature_df[column_order]
    rgb_pca_np = rgb_pca_df.to_numpy()
    rgb_matrix = rgb_pca_np[:, 0:3].astype(np.uint8)
    rgb_matrix = rgb_matrix.reshape((rgb_matrix.shape[0], 1, rgb_matrix.shape[1]))
    expected_pca_array = rgb_pca_np[:, 3].astype(np.uint8).reshape(rgb_pca_np.shape[0], 1, 1)

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_channels(
        rgb_matrix,
        monochrome_pca_components, monochrome_pca_mean,
        monochrome_pca_min, monochrome_pca_max)

    # assert np.array_equal(scaled_pca, expected_pca_array)

    mismatch_count = 0
    for index, (actual, expected) in enumerate(zip(scaled_pca, expected_pca_array)):
        if not np.isclose(actual, expected, atol=1):
            print(f'{index} {actual} != {expected} (R,G,B = {rgb_matrix[index, 0, 0]}, {rgb_matrix[index, 0, 1]},'
                  f' {rgb_matrix[index, 0, 2]})')
            mismatch_count += 1

    # assert np.array_equal(vegetation_label_array, expected_vegetation_label_array)
    assert mismatch_count == 0


def test_neural_network_creates_bright_pca_full_training_set():
    features_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_inputs.csv'

    feature_df = pd.read_csv(str(features_path), header=0, delimiter=',')
    feature_df.drop(columns=['pca_mono', 'Ir', 'pca_colour_1', 'pca_colour_2', 'pca_colour_3'], inplace=True)  # leaves: R,G,B,pca_mono
    column_order = ['B', 'G', 'R', 'pca_bright_1', 'pca_bright_2', 'pca_bright_3']
    bgr_pca_df = feature_df[column_order]
    bgr_pca_np = bgr_pca_df.to_numpy()
    bgr_image = bgr_pca_np[:, 0:3].astype(np.uint8)
    bgr_image = bgr_image.reshape((bgr_image.shape[0], 1, bgr_image.shape[1]))

    expected_pca_array = bgr_pca_np[:, 3:6].astype(np.uint8).reshape(bgr_pca_np.shape[0], 1, 3)

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    value_array = hsv_image[:, :, 2]
    lightness_array = lab_image[:, :, 0]

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_normalised_channels(
        NeuralNetworkClassifier.concatenate_channels(
            [bgr_image[:, :, 2], bgr_image[:, :, 1], bgr_image[:, :, 0], value_array, lightness_array]),
        brightness_pca_components, brightness_pca_mean,
        brightness_pca_min, brightness_pca_max, brightness_pca_inputs_min, brightness_pca_inputs_max)

    # assert np.array_equal(scaled_pca, expected_pca_array)

    mismatch_count = 0
    for index, (actual, expected) in enumerate(zip(scaled_pca, expected_pca_array)):
        if not np.allclose(actual, expected, atol=1):
            print(f'{index} {actual} != {expected} (R,G,B = {bgr_image[index, 0, 2]}, {bgr_image[index, 0, 1]},'
                  f' {bgr_image[index, 0, 0]})')
            mismatch_count += 1

    assert mismatch_count == 0


def test_neural_network_creates_colour_pca_full_training_set():
    features_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_inputs.csv'

    feature_df = pd.read_csv(str(features_path), header=0, delimiter=',')
    feature_df.drop(columns=['pca_mono', 'Ir', 'pca_bright_1', 'pca_bright_2', 'pca_bright_3'], inplace=True)  # leaves: R,G,B,pca_mono
    column_order = ['B', 'G', 'R', 'pca_colour_1', 'pca_colour_2', 'pca_colour_3']
    bgr_pca_df = feature_df[column_order]
    bgr_pca_np = bgr_pca_df.to_numpy()
    bgr_image = bgr_pca_np[:, 0:3].astype(np.uint8)
    bgr_image = bgr_image.reshape((bgr_image.shape[0], 1, bgr_image.shape[1]))

    expected_pca_array = bgr_pca_np[:, 3:6].astype(np.uint8).reshape(bgr_pca_np.shape[0], 1, 3)

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    hue_array = hsv_image[:, :, 0]
    a_array = lab_image[:, :, 1]
    b_array = lab_image[:, :, 2]

    scaled_pca = NeuralNetworkClassifier.generate_8bit_pca_from_n_normalised_channels(
        NeuralNetworkClassifier.concatenate_channels(
            [bgr_image[:, :, 2], bgr_image[:, :, 1], bgr_image[:, :, 0], hue_array, a_array, b_array]),
        colour_pca_components, colour_pca_mean, colour_pca_min, colour_pca_max, colour_pca_inputs_min,
        colour_pca_inputs_max)

    # assert np.array_equal(scaled_pca, expected_pca_array)

    mismatch_count = 0
    for index, (actual, expected) in enumerate(zip(scaled_pca, expected_pca_array)):
        if not np.allclose(actual, expected, atol=1):
            print(f'{index} {actual} != {expected} (R,G,B = {bgr_image[index, 0, 2]}, {bgr_image[index, 0, 1]},'
                  f' {bgr_image[index, 0, 0]})')
            mismatch_count += 1

    assert mismatch_count == 0


def test_neural_network_test_underlying_nn_full_training_set():
    nn_state_path = PurePath(os.path.realpath(__file__)).parent.parent / 'green_spaces' / 'config' / '12_8_4_all_ANN.h5'
    nn = load_model(str(nn_state_path))

    features_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_inputs.csv'
    features_df = pd.read_csv(str(features_path), header=0, delimiter=',')
    # column_order = ['B', 'G', 'R', 'pca_colour_1', 'pca_colour_2', 'pca_colour_3']
    # bgr_pca_df = feature_df[column_order]
    features_np = features_df.to_numpy()

    expected_classes_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_output_classes.csv'
    expected_classes_df = pd.read_csv(str(expected_classes_path), header=0, delimiter=',')
    expected_classes_np = expected_classes_df.to_numpy()
    expected_classes_np = expected_classes_np.reshape((expected_classes_np.shape[0]))

    predicted_classes = nn.predict_classes(features_np)

    # assert predicted_classes == expected_classes_np
    mismatch_count = 0
    for index, (actual, expected) in enumerate(zip(predicted_classes, expected_classes_np)):
        if actual != expected:
            print(f'{index} {actual} != {expected} ({features_np[index, :]})')
            mismatch_count += 1

    # assert np.array_equal(vegetation_label_array, expected_vegetation_label_array)
    assert mismatch_count == 0


def test_neural_network_full_training_set():
    features_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_inputs.csv'
    target_path = PurePath(os.path.realpath(__file__)).parent / 'data' / 'validation_output_classes.csv'

    feature_df = pd.read_csv(str(features_path), header=0, delimiter=',')
    feature_df.drop(columns=['pca_mono', 'pca_bright_1', 'pca_bright_2', 'pca_bright_3', 'pca_colour_1', 'pca_colour_2',
                             'pca_colour_3'], inplace=True)  # leaves: R,G,B,Ir
    column_order = ['B', 'G', 'R', 'Ir']
    bgri_df = feature_df[column_order]
    bgri_matrix = bgri_df.to_numpy().astype(np.uint8)

    target_df = pd.read_csv(str(target_path), header=0, delimiter=',')
    # target_df.drop(columns=['ID'], inplace=True)  # leaves: labelB,labelG,labelR,labelNone
    targets_matrix = target_df.to_numpy()

    bgri_image = bgri_matrix.reshape((bgri_matrix.shape[0], 1, bgri_matrix.shape[1]))
    # target_array = np.add(targets_matrix[:, 0], targets_matrix[:, 1])
    target_array = np.where(targets_matrix[:, 0] < 2, 1, 0)
    expected_vegetation_label_array = target_array.reshape((target_array.shape[0], 1)).astype(np.uint8)

    nnc = NeuralNetworkClassifier({"short_name": "nn",
                                   "full_name": "Neural network vegetation classifier",
                                   "nn_state_path": "12_8_4_all_ANN.h5",

                                   "monochrome_pca_components_path": "pca_mono.pkl",
                                   "monochrome_pca_mean_path": "pca_mono_mean.pkl",
                                   "monochrome_pca_min": monochrome_pca_min,
                                   "monochrome_pca_max": monochrome_pca_max,

                                   "brightness_pca_components_path": "pca_bright.pkl",
                                   "brightness_pca_mean_path": "pca_bright_mean.pkl",
                                   "brightness_pca_inputs_min": brightness_pca_inputs_min,
                                   "brightness_pca_inputs_max": brightness_pca_inputs_max,
                                   "brightness_pca_min": brightness_pca_min,
                                   "brightness_pca_max": brightness_pca_max,

                                   "colour_pca_components_path": "pca_colour.pkl",
                                   "colour_pca_mean_path": "pca_colour_mean.pkl",
                                   "colour_pca_inputs_min": colour_pca_inputs_min,
                                   "colour_pca_inputs_max": colour_pca_inputs_max,
                                   "colour_pca_min": colour_pca_min,
                                   "colour_pca_max": colour_pca_max,
                                   },
                                  load_pickle_fn=load_pca_pickle)

    vegetation_label_array = nnc.index(bgri_image)

    mismatch_count = 0
    for index, (actual, expected) in enumerate(zip(vegetation_label_array, expected_vegetation_label_array)):
        if actual != expected:
            print(f'{index} {actual} != {expected} (B,G,R,Ir = {bgri_matrix[index, 0]}, {bgri_matrix[index, 1]},'
                  f' {bgri_matrix[index, 2]}, {bgri_matrix[index, 3]})')
            mismatch_count += 1

    # assert np.array_equal(vegetation_label_array, expected_vegetation_label_array)
    # assert mismatch_count == 0
