import os

import numpy as np
import pytest
import shapely.wkt

from green_spaces.image_loaders import OrdnanceSurveyMapLoader, OrdnanceSurveyMapLoaderWithInfraRed

actual_image_loader_calls = 0

latitude_longitude_coord_system = "urn:ogc:def:crs:OGC:1.3:CRS84"
eastings_northings_coord_system = "urn:ogc:def:crs:EPSG::27700"


def extract_poly_coords(geom):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return {'exterior_coords': exterior_coords,
            'interior_coords': interior_coords}


def test_OrdnanceSurveyMapLoader_raises_error_with_unknown_coordinate_system():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 1000000
    crs_name = "unknown system"

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    with pytest.raises(ValueError) as e_info:
        OrdnanceSurveyMapLoader(loader_config, crs_name, primary_cache_size, 0, None)
    assert e_info.value.args[0] == f'crs_name="{crs_name}" is unsupported'


def test_OrdnanceSurveyMapLoader_supports_eastings_northings():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 1000000

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, eastings_northings_coord_system, primary_cache_size, 0, None)

    eastings_northings_geometry = shapely.wkt.loads('MULTIPOLYGON (((367220.85 170316.2, 367220.55 170316.3, '
                                                    '367205.423 170315.39, 367220.85 170316.2)))')
    expected_tile_geometry = shapely.wkt.loads('MULTIPOLYGON (((367.22085 170.3162, 367.22055 170.3163, '
                                               '367.205423 170.31539, 367.22085 170.3162)))')
    tile_geometry = loader.calculate_tile_geometry(eastings_northings_geometry)

    np.testing.assert_almost_equal(
        extract_poly_coords(tile_geometry)['exterior_coords'],
        extract_poly_coords(expected_tile_geometry)['exterior_coords'],
        decimal=5
    )


def test_OrdnanceSurveyMapLoader_supports_latitude_longitude():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 1000000

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, None)

    eastings_northings_geometry = shapely.wkt.loads('MULTIPOLYGON (((-2.472899455869044 51.430893105324593, '
                                                    '-2.472903780332496 51.430893987034189, '
                                                    '-2.473121281324169 51.430884926567657, '
                                                    '-2.472899455869044 51.430893105324593)))')
    expected_tile_geometry = shapely.wkt.loads('MULTIPOLYGON (((367.22085 170.3162, 367.22055 170.3163, '
                                               '367.205423 170.31539, 367.22085 170.3162)))')
    tile_geometry = loader.calculate_tile_geometry(eastings_northings_geometry)

    np.testing.assert_almost_equal(
        extract_poly_coords(tile_geometry)['exterior_coords'],
        extract_poly_coords(expected_tile_geometry)['exterior_coords'],
        decimal=5
    )


def test_OrdnanceSurveyMapLoader_build_tile_file_name():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 1000000

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, None)

    eastings = 702
    northings = 345
    expected_path = os.path.join('TH', 'TH04', 'TH0245.jpg')

    actual_path = loader.build_tile_file_name(eastings, northings)

    assert expected_path == actual_path


def test_OrdnanceSurveyMapLoader_retrieve_image_loads_rgb_as_bgr_red():
    expected_image = np.array([
        [[0, 0, 255], [0, 0, 255]],
        [[0, 0, 255], [0, 0, 255]]
    ], dtype=np.uint8)
    loader_config = {'folder': 'tests/data/images_RGB', 'tile_size': 2, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('red.png')
    np.testing.assert_equal(actual_image, expected_image)


def test_OrdnanceSurveyMapLoader_retrieve_image_loads_rgb_as_bgr_green():
    expected_image = np.array([
        [[0, 255, 0], [0, 255, 0]],
        [[0, 255, 0], [0, 255, 0]]
    ], dtype=np.uint8)
    loader_config = {'folder': 'tests/data/images_RGB', 'tile_size': 2, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('green.png')
    np.testing.assert_equal(actual_image, expected_image)


def test_OrdnanceSurveyMapLoader_retrieve_image_loads_rgb_as_bgr_blue():
    expected_image = np.array([
        [[255, 0, 0], [255, 0, 0]],
        [[255, 0, 0], [255, 0, 0]]
    ], dtype=np.uint8)
    loader_config = {'folder': 'tests/data/images_RGB', 'tile_size': 2, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('blue.png')
    np.testing.assert_equal(actual_image, expected_image)


def test_OrdnanceSurveyMapLoader_retrieve_image_resizes_to_config():
    expected_image = np.array([
        [[255, 0, 0], [255, 0, 0], [255, 0, 0]],
        [[255, 0, 0], [255, 0, 0], [255, 0, 0]],
        [[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    ], dtype=np.uint8)
    loader_config = {'folder': 'tests/data/images_RGB', 'tile_size': 3, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('blue.png')
    np.testing.assert_equal(actual_image, expected_image)
    assert loader.warnings == ['Image "tests/data/images_RGB/blue.png" is sized (2, 2, 3) rather than (3, 3, 3))']


def test_OrdnanceSurveyMapLoaderWithInfraRed_retrieve_image_loads_rgb_cir_as_bgrir_blue():
    expected_image = np.array([
        [[255, 0, 0, 0], [255, 0, 0, 0]],
        [[255, 0, 0, 0], [255, 0, 0, 0]]
    ], dtype=np.uint8)
    loader_config = {'folder_RGB': 'tests/data/images_RGB',
                     'folder_CIR': 'tests/data/images_CIR',
                     'final_tile_size': 2, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('blue.png')
    np.testing.assert_equal(actual_image, expected_image)


def test_OrdnanceSurveyMapLoaderWithInfraRed_retrieve_image_loads_rgb_cir_as_bgrir_blue_plus_ir():
    expected_image = np.array([
        [[255, 0, 0, 255], [255, 0, 0, 255]],
        [[255, 0, 0, 255], [255, 0, 0, 255]]
    ], dtype=np.uint8)
    loader_config = {'folder_RGB': 'tests/data/images_RGB',
                     'folder_CIR': 'tests/data/images_CIR',
                     'final_tile_size': 2, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('blue+ir.png')
    np.testing.assert_equal(actual_image, expected_image)


def test_OrdnanceSurveyMapLoaderWithInfraRed_retrieve_image_resizes_to_config():
    expected_image = np.array([
        [[255, 0, 0, 255]]
    ], dtype=np.uint8)
    loader_config = {'folder_RGB': 'tests/data/images_RGB',
                     'folder_CIR': 'tests/data/images_CIR',
                     'final_tile_size': 1, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, 0, 0)
    actual_image = loader.retrieve_image('blue+ir.png')
    np.testing.assert_equal(actual_image, expected_image)
    assert loader.warnings == ['Image "tests/data/images_RGB/blue+ir.png" is sized (2, 2, 3) rather than (1, 1, 3))']
    # Note that IR is mal-sized but re-sized to match RGB and hence final image is required size
    # (as RGB is re-sized first)


def test_OrdnanceSurveyMapLoader_retrieve_image_caches():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 2000000
    global actual_image_loader_calls
    actual_image_loader_calls = 0
    expected_image_loader_calls = 2
    image1_file_name = 'some file name 1'
    image2_file_name = 'some file name 2'

    def image_loader(_):
        global actual_image_loader_calls
        actual_image_loader_calls += 1
        return np.zeros((tile_size, tile_size, 3))

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader)

    img = loader.retrieve_image(image1_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size, tile_size, 3) == img.shape

    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoader_retrieve_image_caches_max_cache_size():
    root_folder_path = 'root'
    tile_size = 500
    primary_cache_size = 1000000
    global actual_image_loader_calls
    actual_image_loader_calls = 0
    expected_image_loader_calls = 3
    image1_file_name = 'some file name 1'
    image2_file_name = 'some file name 2'

    def image_loader(_):
        global actual_image_loader_calls
        actual_image_loader_calls += 1
        return np.zeros((tile_size, tile_size, 3))

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader)

    img = loader.retrieve_image(image1_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size, tile_size, 3) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size, tile_size, 3) == img.shape

    assert expected_image_loader_calls == actual_image_loader_calls


@pytest.fixture
def create_image_loader():
    root_folder_path = 'root'
    tile_size = 100
    primary_cache_size = 2
    global actual_image_loader_calls
    actual_image_loader_calls = []

    def image_loader(image_file_name):
        global actual_image_loader_calls
        actual_image_loader_calls.append(image_file_name)

        last_3_cell_chars = image_file_name[-7:-4]
        fill_value = int(last_3_cell_chars) % 256
        return np.full(shape=(tile_size, tile_size, 3), fill_value=fill_value).astype(np.uint8)

    loader_config = {'folder': root_folder_path, 'tile_size': tile_size, 'name': 'test_OS'}
    return OrdnanceSurveyMapLoader(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader), root_folder_path, tile_size


def test_OrdnanceSurveyMapLoader_download_image_1x1(create_image_loader):
    loader, root_folder_path, tile_size = create_image_loader

    max_tile_e = 401
    max_tile_n = 126
    min_tile_e = 401
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path, 'SU', 'SU02', 'SU0126.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size, tile_size, 3) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoader_download_image_1x2(create_image_loader):
    loader, root_folder_path, tile_size = create_image_loader

    max_tile_e = 401
    max_tile_n = 127
    min_tile_e = 401
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path, 'SU', 'SU02', 'SU0126.jpg'),
        os.path.join(root_folder_path, 'SU', 'SU02', 'SU0127.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size * 2, tile_size, 3) == img.shape
    assert np.array_equal(img[0:tile_size, :, :], np.full(shape=(tile_size, tile_size, 3), fill_value=127))
    assert np.array_equal(img[tile_size:tile_size * 2, :, :], np.full(shape=(tile_size, tile_size, 3), fill_value=126))
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoader_download_image_2x1(create_image_loader):
    loader, root_folder_path, tile_size = create_image_loader

    max_tile_e = 401
    max_tile_n = 126
    min_tile_e = 400
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path, 'SU', 'SU02', 'SU0026.jpg'),
        os.path.join(root_folder_path, 'SU', 'SU02', 'SU0126.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size, tile_size * 2, 3) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoader_download_image_2x2_crosses_images(create_image_loader):
    loader, root_folder_path, tile_size = create_image_loader

    max_tile_e = 400
    max_tile_n = 100
    min_tile_e = 399
    min_tile_n = 99
    expected_image_loader_calls = [
        os.path.join(root_folder_path, 'SY', 'SY99', 'SY9999.jpg'),
        os.path.join(root_folder_path, 'SZ', 'SZ09', 'SZ0099.jpg'),
        os.path.join(root_folder_path, 'ST', 'ST90', 'ST9900.jpg'),
        os.path.join(root_folder_path, 'SU', 'SU00', 'SU0000.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size * 2, tile_size * 2, 3) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoader_download_image_2x2_crosses_letter_tiles(create_image_loader):
    loader, root_folder_path, tile_size = create_image_loader

    max_tile_e = 500
    max_tile_n = 500
    min_tile_e = 499
    min_tile_n = 499
    expected_image_loader_calls = [
        os.path.join(root_folder_path, 'SE', 'SE99', 'SE9999.jpg'),
        os.path.join(root_folder_path, 'TA', 'TA09', 'TA0099.jpg'),
        os.path.join(root_folder_path, 'NZ', 'NZ90', 'NZ9900.jpg'),
        os.path.join(root_folder_path, 'OV', 'OV00', 'OV0000.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size * 2, tile_size * 2, 3) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_raises_error_with_unknown_coordinate_system():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'ir'
    tile_size_ir = 50
    primary_cache_size = 1000000
    crs_name = "unknown system"

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': tile_size_ir, 'name': 'test_OS'}
    with pytest.raises(ValueError) as e_info:
        OrdnanceSurveyMapLoaderWithInfraRed(loader_config, crs_name, primary_cache_size, 0, None)
    assert e_info.value.args[0] == f'crs_name="{crs_name}" is unsupported'


def test_OrdnanceSurveyMapLoaderWithInfraRed_supports_eastings_northings():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'ir'
    tile_size_ir = 50
    primary_cache_size = 1000000

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': tile_size_ir, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, eastings_northings_coord_system, primary_cache_size, 0, None)

    eastings_northings_geometry = shapely.wkt.loads('MULTIPOLYGON (((367220.85 170316.2, 367220.55 170316.3, 367205.423 170315.39, 367220.85 170316.2)))')
    expected_tile_geometry = shapely.wkt.loads('MULTIPOLYGON (((367.22085 170.3162, 367.22055 170.3163, 367.205423 170.31539, 367.22085 170.3162)))')
    tile_geometry = loader.calculate_tile_geometry(eastings_northings_geometry)
    np.testing.assert_almost_equal(
        extract_poly_coords(tile_geometry)['exterior_coords'],
        extract_poly_coords(expected_tile_geometry)['exterior_coords'],
        decimal=5
    )


def test_OrdnanceSurveyMapLoaderWithInfraRed_supports_latitude_longitude():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'ir'
    tile_size_ir = 50
    primary_cache_size = 1000000

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': tile_size_ir, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, None)

    eastings_northings_geometry = shapely.wkt.loads('MULTIPOLYGON (((-2.472899455869044 51.430893105324593, '
                                                    '-2.472903780332496 51.430893987034189, '
                                                    '-2.473121281324169 51.430884926567657, '
                                                    '-2.472899455869044 51.430893105324593)))')
    expected_tile_geometry = shapely.wkt.loads('MULTIPOLYGON (((367.22085 170.3162, 367.22055 170.3163, '
                                               '367.205423 170.31539, 367.22085 170.3162)))')
    tile_geometry = loader.calculate_tile_geometry(eastings_northings_geometry)

    np.testing.assert_almost_equal(
        extract_poly_coords(tile_geometry)['exterior_coords'],
        extract_poly_coords(expected_tile_geometry)['exterior_coords'],
        decimal=5
    )


def test_OrdnanceSurveyMapLoaderWithInfraRed_build_tile_file_name():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'ir'
    tile_size_ir = 50
    primary_cache_size = 1000000

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': tile_size_ir, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, None)

    eastings = 702
    northings = 345
    expected_path = os.path.join('TH', 'TH04', 'TH0245.jpg')

    actual_path = loader.build_tile_file_name(eastings, northings)

    assert expected_path == actual_path


def test_OrdnanceSurveyMapLoaderWithInfraRed_retrieve_image_caches():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'cir'
    final_tile_size = 100
    tile_size_rgb = 100
    tile_size_ir = 50
    primary_cache_size = 2000000

    global actual_image_loader_calls
    actual_image_loader_calls = 0
    expected_image_loader_calls = 4
    image1_file_name = 'some file name 1'
    image2_file_name = 'some file name 2'

    def image_loader(file_name):
        global actual_image_loader_calls
        actual_image_loader_calls += 1
        if file_name.startswith(root_folder_path_rgb):
            return np.zeros((tile_size_rgb, tile_size_rgb, 3))
        elif file_name.startswith(root_folder_path_cir):
            return np.ones((tile_size_ir, tile_size_ir, 3))
        else:
            return None

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': final_tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader)

    img = loader.retrieve_image(image1_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape

    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_retrieve_image_caches_max_cache_size():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'cir'
    final_tile_size = 100
    tile_size_rgb = 100
    tile_size_ir = 25
    primary_cache_size = 50000

    global actual_image_loader_calls
    actual_image_loader_calls = 0
    expected_image_loader_calls = 6
    image1_file_name = 'some file name 1'
    image2_file_name = 'some file name 2'

    def image_loader(file_name):
        global actual_image_loader_calls
        actual_image_loader_calls += 1
        if file_name.startswith(root_folder_path_rgb):
            return np.zeros((tile_size_rgb, tile_size_rgb, 3))
        elif file_name.startswith(root_folder_path_cir):
            return np.ones((tile_size_ir, tile_size_ir, 3))
        else:
            return None

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': final_tile_size, 'name': 'test_OS'}
    loader = OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader)

    img = loader.retrieve_image(image1_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image2_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    img = loader.retrieve_image(image1_file_name)
    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape

    assert expected_image_loader_calls == actual_image_loader_calls


@pytest.fixture
def create_RGBIr_image_loader():
    root_folder_path_rgb = 'rgb'
    root_folder_path_cir = 'cir'
    final_tile_size = 100
    tile_size_rgb = 100
    tile_size_ir = 25
    primary_cache_size = 200000
    global actual_image_loader_calls
    actual_image_loader_calls = []

    def image_loader(image_file_name):
        global actual_image_loader_calls
        actual_image_loader_calls.append(image_file_name)

        if image_file_name.startswith(root_folder_path_rgb):
            last_3_cell_chars = image_file_name[-7:-4]
            fill_value = int(last_3_cell_chars) % 256
            return np.full(shape=(tile_size_rgb, tile_size_rgb, 3), fill_value=fill_value).astype(np.uint8)

        elif image_file_name.startswith(root_folder_path_cir):
            last_3_cell_chars = image_file_name[-7:-4]
            fill_value = int(last_3_cell_chars) % 256
            return np.full(shape=(tile_size_ir, tile_size_ir, 3), fill_value=fill_value).astype(np.uint8)

        else:
            return None

    loader_config = {'folder_RGB': root_folder_path_rgb, 'folder_CIR': root_folder_path_cir,
                     'final_tile_size': final_tile_size, 'name': 'test_OS'}
    return OrdnanceSurveyMapLoaderWithInfraRed(loader_config, latitude_longitude_coord_system, primary_cache_size, 0, image_loader), \
           root_folder_path_rgb, tile_size_rgb, root_folder_path_cir, tile_size_ir


def test_OrdnanceSurveyMapLoaderWithInfraRed_download_image_1x1(create_RGBIr_image_loader):
    loader, root_folder_path_rgb, tile_size_rgb, root_folder_path_ir, tile_size_ir = create_RGBIr_image_loader

    max_tile_e = 401
    max_tile_n = 126
    min_tile_e = 401
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path_rgb, 'SU', 'SU02', 'SU0126.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU02', 'SU0126.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size_rgb, tile_size_rgb, 4) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_download_image_1x2(create_RGBIr_image_loader):
    loader, root_folder_path_rgb, tile_size_rgb, root_folder_path_ir, tile_size_ir = create_RGBIr_image_loader

    max_tile_e = 401
    max_tile_n = 127
    min_tile_e = 401
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path_rgb, 'SU', 'SU02', 'SU0126.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU02', 'SU0126.jpg'),
        os.path.join(root_folder_path_rgb, 'SU', 'SU02', 'SU0127.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU02', 'SU0127.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size_rgb * 2, tile_size_rgb, 4) == img.shape
    assert np.array_equal(img[0:tile_size_rgb, :, :], np.full(shape=(tile_size_rgb, tile_size_rgb, 4), fill_value=127))
    assert np.array_equal(img[tile_size_rgb:tile_size_rgb * 2, :, :],
                          np.full(shape=(tile_size_rgb, tile_size_rgb, 4), fill_value=126))
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_download_image_2x1(create_RGBIr_image_loader):
    loader, root_folder_path_rgb, tile_size_rgb, root_folder_path_ir, tile_size_ir = create_RGBIr_image_loader

    max_tile_e = 401
    max_tile_n = 126
    min_tile_e = 400
    min_tile_n = 126
    expected_image_loader_calls = [
        os.path.join(root_folder_path_rgb, 'SU', 'SU02', 'SU0026.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU02', 'SU0026.jpg'),
        os.path.join(root_folder_path_rgb, 'SU', 'SU02', 'SU0126.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU02', 'SU0126.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size_rgb, tile_size_rgb * 2, 4) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_download_image_2x2_crosses_images(create_RGBIr_image_loader):
    loader, root_folder_path_rgb, tile_size_rgb, root_folder_path_ir, tile_size_ir = create_RGBIr_image_loader

    max_tile_e = 400
    max_tile_n = 100
    min_tile_e = 399
    min_tile_n = 99
    expected_image_loader_calls = [
        os.path.join(root_folder_path_rgb, 'SY', 'SY99', 'SY9999.jpg'),
        os.path.join(root_folder_path_ir, 'SY', 'SY99', 'SY9999.jpg'),
        os.path.join(root_folder_path_rgb, 'SZ', 'SZ09', 'SZ0099.jpg'),
        os.path.join(root_folder_path_ir, 'SZ', 'SZ09', 'SZ0099.jpg'),
        os.path.join(root_folder_path_rgb, 'ST', 'ST90', 'ST9900.jpg'),
        os.path.join(root_folder_path_ir, 'ST', 'ST90', 'ST9900.jpg'),
        os.path.join(root_folder_path_rgb, 'SU', 'SU00', 'SU0000.jpg'),
        os.path.join(root_folder_path_ir, 'SU', 'SU00', 'SU0000.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size_rgb * 2, tile_size_rgb * 2, 4) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls


def test_OrdnanceSurveyMapLoaderWithInfraRed_download_image_2x2_crosses_letter_tiles(create_RGBIr_image_loader):
    loader, root_folder_path_rgb, tile_size_rgb, root_folder_path_ir, tile_size_ir = create_RGBIr_image_loader

    max_tile_e = 500
    max_tile_n = 500
    min_tile_e = 499
    min_tile_n = 499
    expected_image_loader_calls = [
        os.path.join(root_folder_path_rgb, 'SE', 'SE99', 'SE9999.jpg'),
        os.path.join(root_folder_path_ir, 'SE', 'SE99', 'SE9999.jpg'),
        os.path.join(root_folder_path_rgb, 'TA', 'TA09', 'TA0099.jpg'),
        os.path.join(root_folder_path_ir, 'TA', 'TA09', 'TA0099.jpg'),
        os.path.join(root_folder_path_rgb, 'NZ', 'NZ90', 'NZ9900.jpg'),
        os.path.join(root_folder_path_ir, 'NZ', 'NZ90', 'NZ9900.jpg'),
        os.path.join(root_folder_path_rgb, 'OV', 'OV00', 'OV0000.jpg'),
        os.path.join(root_folder_path_ir, 'OV', 'OV00', 'OV0000.jpg'),
    ]

    img = loader.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

    assert (tile_size_rgb * 2, tile_size_rgb * 2, 4) == img.shape
    assert expected_image_loader_calls == actual_image_loader_calls
