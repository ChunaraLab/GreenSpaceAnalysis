import operator
import urllib.request
from math import floor
from os import makedirs
from os import path
import os
from urllib.parse import urlparse
from functools import partial

import cv2
import numpy as np
import rasterio
from cachetools import cachedmethod, LRUCache
from rasterio.features import geometry_mask
from shapely.affinity import affine_transform
from shapely.ops import transform

from green_spaces.coordinate_transforms import lat_long_to_web_mercator_tile_yx, \
    tile_eastings_and_northings_to_tile_code, lat_long_to_fractional_tile_eastings_and_nothings


def url_to_offline_file_name(image_url, zoom):
    url = urlparse(image_url)
    url_path = url.path[1:]
    if len(url.query) == 0:
        image_file_name = path.join(f'cache-zoom{zoom}', url.hostname, url_path)
    else:
        url_args = url.query.split('&')
        image_file_name = path.join(f'cache-zoom{zoom}', url.hostname, url_path, *url_args)
    return image_file_name


class WebMercatorMapLoader(object):
    __tile_size = 256

    def __init__(self, loader_config, crs_name, primary_cache_size, enable_secondary_cache):
        self.__name = loader_config['name']
        self.__tile_loader_url = loader_config['url']
        self.__zoom = loader_config['zoom']
        self.__enable_secondary_cache = enable_secondary_cache
        primary_cache_size_in_tiles = floor(
            primary_cache_size / (self.__tile_size * self.__tile_size * 3))
        self.image_cache = LRUCache(maxsize=primary_cache_size_in_tiles)

        self.__image_requests = 0
        self.__image_cache_misses = 0

        if crs_name == "urn:ogc:def:crs:OGC:1.3:CRS84":  # latitude, longitude
            def lat_long_to_tile_yx_func(x, y, z=None):
                tile_y, tile_x = lat_long_to_web_mercator_tile_yx(self.__zoom, y, x)
                return tuple(filter(None, [tile_y, tile_x, z]))
            self.__transform_func = partial(transform, func=lat_long_to_tile_yx_func)

        else:
            raise ValueError(f'crs_name="{crs_name}" is unsupported')

    @cachedmethod(operator.attrgetter('image_cache'))
    def image_retrieval(self, image_url):
        # print(f'Fetching {image_url}')
        self.__image_cache_misses += 1
        image_file_name = None

        if self.__enable_secondary_cache:
            image_file_name = url_to_offline_file_name(image_url, self.__zoom)
            if not image_file_name.endswith('.png'):
                image_file_name += '.png'

            if path.isfile(image_file_name):
                image_cv2 = cv2.imread(image_file_name)
                return image_cv2

        req = urllib.request.Request(
            image_url,
            data=None
        )
        with urllib.request.urlopen(req) as img_request_response:
            image_raw_data = img_request_response.read()
            image_raw_array = np.fromstring(image_raw_data, dtype='uint8')
            image_cv2 = cv2.imdecode(image_raw_array, cv2.IMREAD_UNCHANGED)

        if self.__enable_secondary_cache:
            image_folder_name = path.dirname(image_file_name)
            makedirs(image_folder_name, exist_ok=True)
            cv2.imwrite(image_file_name, image_cv2)

        return image_cv2[:, :, 0:3]

    def build_tile_loader_url(self, x, y):
        url = self.__tile_loader_url.replace('{x}', str(x)).replace('{y}', str(y)).replace('{zoom}', str(self.__zoom))
        return url

    def download_image(self, max_tile_y, max_tile_x, min_tile_y, min_tile_x):
        width = (max_tile_x - min_tile_x + 1) * self.__tile_size
        height = (max_tile_y - min_tile_y + 1) * self.__tile_size

        overall_tile = np.zeros((height, width, 3), np.uint8)
        for y in range(min_tile_y, max_tile_y + 1):
            for x in range(min_tile_x, max_tile_x + 1):
                image_tile_url = self.build_tile_loader_url(x, y)

                self.__image_requests += 1
                image_tile = self.image_retrieval(image_tile_url)

                x_offset = (x - min_tile_x) * self.__tile_size
                y_offset = (y - min_tile_y) * self.__tile_size

                overall_tile[
                    y_offset:(y_offset + self.__tile_size),
                    x_offset:(x_offset + self.__tile_size),
                    :] = image_tile

        return overall_tile

    def calculate_tile_geometry(self, geometry):
        tile_geometry = self.__transform_func(geometry)
        return tile_geometry

    def get_image_and_mask(self, tile_geometry, debug_base_file_name=None):

        surface_area = tile_geometry.area

        min_tile_y = int(tile_geometry.bounds[0])
        min_tile_x = int(tile_geometry.bounds[1])
        max_tile_y = int(tile_geometry.bounds[2])
        max_tile_x = int(tile_geometry.bounds[3])

        image_bgr = self.download_image(max_tile_y, max_tile_x, min_tile_y, min_tile_x)

        # [a, b, d, e, xoff, yoff]
        # x' = a * x + b * y + xoff
        # y' = d * x + e * y + yoff
        m = [0, self.__tile_size, self.__tile_size, 0.0, -min_tile_x * self.__tile_size, -min_tile_y * self.__tile_size]
        affine_geometry = affine_transform(tile_geometry, m)

        min_y = floor(affine_geometry.bounds[1])
        min_x = floor(affine_geometry.bounds[0])
        max_y = floor(affine_geometry.bounds[3])
        max_x = floor(affine_geometry.bounds[2])

        affine = rasterio.Affine(1, 0, min_x, 0, 1, min_y)

        pixels_within_geometry = geometry_mask([affine_geometry],
                                               (max_y - min_y + 1, max_x - min_x + 1),
                                               affine, invert=True)

        image_bgr_cropped = image_bgr[min_y:max_y + 1, min_x:max_x + 1, :]

        tile_file_name = None

        if debug_base_file_name is not None:
            centre_point = tile_geometry.centroid
            centre_tile_x = int(centre_point.x)
            centre_tile_y = int(centre_point.y)
            centre_pixel_x = int((centre_point.x - centre_tile_x) * self.__tile_size)
            centre_pixel_y = int((centre_point.y - centre_tile_y) * self.__tile_size)
            tile_code = f'{centre_tile_x}_{centre_pixel_x}={centre_tile_y}_{centre_pixel_y}'
            tile_file_name = debug_base_file_name + '-' + tile_code

        return image_bgr_cropped, pixels_within_geometry, surface_area, tile_file_name

    def report_usage(self):
        ## Changed here !!! We do not use their image loading using masks
        # print(f'Number of map tile requests: {self.image_requests:,}')
        # print(f'Number of map tile cache hits vs misses: '
        #       f'{self.image_cache_hits:,} vs {self.image_cache_misses:,}')
        print()

    @property
    def name(self):
        return self.__name

    @property
    def image_cache_hits(self):
        return self.__image_requests - self.__image_cache_misses

    @property
    def image_requests(self):
        return self.__image_requests

    @property
    def image_cache_misses(self):
        return self.__image_cache_misses

    @property
    def statistics_report(self):
        return f'{self.image_cache_hits:,} cached, {self.image_cache_misses:,} missed;' \
            f' hit rate {self.image_cache_hits / self.image_requests * 100.0:.1f}%'


class OrdnanceSurveyMapLoader(object):

    def __init__(self, loader_config, crs_name, primary_cache_size, enable_secondary_cache, image_loader=cv2.imread):
        self.__name = loader_config['name']
        # self.__tile_root_folder_path = loader_config['folder']
        self.__tile_root_folder_path = '/content/drive/MyDrive/green-spaces/Images'
        self.__tile_size = loader_config['tile_size']
        self.__image_reader = image_loader
        self.warnings = []
        primary_cache_size_in_tiles = floor(primary_cache_size / (self.__tile_size * self.__tile_size * 3))
        self.image_cache = LRUCache(maxsize=primary_cache_size_in_tiles)

        self.__image_requests = 0
        self.__image_cache_misses = 0
        
        print("*&*&*&*&*&*&*&*&*(^#*&%&%*&#$^*&#$^(")
        print(loader_config)

        if crs_name == "urn:ogc:def:crs:OGC:1.3:CRS84":  # latitude, longitude
            def lat_long_to_tile_eastings_northings(x, y, z=None):
                eastings, northings = lat_long_to_fractional_tile_eastings_and_nothings(y, x)
                return tuple(filter(None, [eastings, northings, z]))
            self.__transform_func = partial(transform, func=lat_long_to_tile_eastings_northings)

        elif crs_name == "urn:ogc:def:crs:EPSG::27700":  # eastings, northings
            def eastings_northings_to_fractional_tile(eastings, northings, z=None):
                return tuple(filter(None, [eastings / 1000, northings / 1000, z]))
            self.__transform_func = partial(transform, func=eastings_northings_to_fractional_tile)

        else:
            raise ValueError(f'crs_name="{crs_name}" is unsupported')

    # @cachedmethod(operator.attrgetter('image_cache'))
    def retrieve_image(self, image_filenames):
        # self.__image_cache_misses += 1

        image_full_file_name = []
        for i in image_filenames:
            image_full_file_name.append(path.join(self.__tile_root_folder_path, i))
        # image_cv2 = self.__image_reader(image_full_file_name)
        image_cv2 = []
        for name in image_full_file_name:
            image_cv2.append(self.__image_reader(name)[:, :, 0:3])

        # if image_cv2.shape != (self.__tile_size, self.__tile_size, 3):
        #     self.warnings.append(f'Image "{image_full_file_name}" is sized {image_cv2.shape}'
        #                          f' rather than ({self.__tile_size}, {self.__tile_size}, 3))')
        #     image_cv2 = cv2.resize(image_cv2, dsize=(self.__tile_size, self.__tile_size))

        return image_cv2, image_filenames

    def build_tile_file_name(self, eastings=None, northings=None):
        # tile_code = tile_eastings_and_northings_to_tile_code(eastings, northings)
        # example file path: NS/NS46/NS4360.jpg
        # file_name = path.join(tile_code[0:2], tile_code[0:3] + tile_code[4], tile_code + '.jpg')
        # file_name = path.join('HP', 'HP01', 'HP0102' + '.png')
        file_names = os.listdir(self.__tile_root_folder_path)
        return file_names

    def download_image(self, max_tile_e, max_tile_n, min_tile_e, min_tile_n):
        image_tile_file_names = self.build_tile_file_name()
        print(image_tile_file_names)
        image_tile, image_filenames = self.retrieve_image(image_tile_file_names)
        print("**** Shape of the loaded images are ****")
        print(np.array(image_tile).shape)
        return image_tile, image_filenames


        width = (max_tile_e - min_tile_e + 1) * self.__tile_size
        height = (max_tile_n - min_tile_n + 1) * self.__tile_size

        overall_tile = np.zeros((height, width, 3), np.uint8)
        for northings in range(min_tile_n, max_tile_n + 1):
            for eastings in range(min_tile_e, max_tile_e + 1):
                image_tile_file_name = self.build_tile_file_name(eastings, northings)

                self.__image_requests += 1
                # print("****************")
                image_tile = self.retrieve_image(image_tile_file_name)
                # print(image_tile.shape)

                e_offset = (eastings - min_tile_e) * self.__tile_size
                n_offset = (max_tile_n - northings) * self.__tile_size

                overall_tile[
                    n_offset:(n_offset + self.__tile_size),
                    e_offset:(e_offset + self.__tile_size),
                    :] = image_tile
        print("******* overall_tile in loader.download_image ********")
        print(overall_tile.shape)
        return overall_tile

    def calculate_tile_geometry(self, geometry):
        tile_geometry = self.__transform_func(geom=geometry)
        return tile_geometry

    def get_image_and_mask(self, tile_geometry, debug_base_file_name=None):

        # Change here!! to your own spatial resolution
        # cm2_to_m2 = 0.01 * 0.01
        km2_to_m2 = 1000.0 * 1000.0
        surface_area_m2 = tile_geometry.area * km2_to_m2


        min_tile_e = int(tile_geometry.bounds[0])
        min_tile_n = int(tile_geometry.bounds[1])
        max_tile_e = int(tile_geometry.bounds[2])
        max_tile_n = int(tile_geometry.bounds[3])


        image_bgr, image_filenames = self.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

        # [a, b, d, e, xoff, yoff]
        # x' = a * x + b * y + xoff
        # y' = d * x + e * y + yoff
        # m = [self.__tile_size, 0, 0, self.__tile_size, -min_tile_e * self.__tile_size, -min_tile_n * self.__tile_size]
        # affine_geometry = affine_transform(tile_geometry, m)

        # min_x = floor(affine_geometry.bounds[0])
        # min_y = floor(affine_geometry.bounds[1])
        # max_x = floor(affine_geometry.bounds[2])
        # max_y = floor(affine_geometry.bounds[3])

        # max_y_vertically_flipped = image_bgr.shape[0] - 1 - min_y
        # min_y_vertically_flipped = image_bgr.shape[0] - 1 - max_y

        # affine = rasterio.Affine(1, 0, min_x, 0, -1, max_y)

        # pixels_within_geometry = geometry_mask([affine_geometry],
        #                                        (max_y_vertically_flipped - min_y_vertically_flipped + 1,
        #                                         max_x - min_x + 1),
        #                                        affine, invert=True)

        # image_bgr_cropped = image_bgr[min_y_vertically_flipped:max_y_vertically_flipped + 1, min_x:max_x + 1, :]

        tile_file_name = None

        # Change here!! Tile name from centre_point
        centre_point = tile_geometry.centroid
        tile_code = tile_eastings_and_northings_to_tile_code(centre_point.x, centre_point.y)
        tile_file_name = tile_code

        # if debug_base_file_name is not None:
        #     centre_point = tile_geometry.centroid
        #     tile_code = tile_eastings_and_northings_to_tile_code(centre_point.x, centre_point.y)
        #     tile_file_name = debug_base_file_name + '-' + tile_code

        print("**** Returned from get_image_and_mask: (image count)(surface area)(tile file name) ****")
        print(len(image_bgr))
        print(surface_area_m2)
        print(tile_file_name)
        return image_bgr, None, surface_area_m2, tile_file_name, image_filenames
        # return image_bgr_cropped, pixels_within_geometry, surface_area_m2, tile_file_name

    def report_usage(self):
        print(f'Number of map tile requests: {self.image_requests:,}')
        print(f'Number of map tile cache hits vs misses: '
              f'{self.image_cache_hits:,} vs {self.image_cache_misses:,}')

    @property
    def name(self):
        return self.__name

    @property
    def image_cache_hits(self):
        return self.__image_requests - self.__image_cache_misses

    @property
    def image_requests(self):
        return self.__image_requests

    @property
    def image_cache_misses(self):
        return self.__image_cache_misses

    @property
    def statistics_report(self):
        return f'{self.image_cache_hits:,} cached, {self.image_cache_misses:,} missed;' \
            f' hit rate {self.image_cache_hits / self.image_requests * 100.0:.1f}%'


class OrdnanceSurveyMapLoaderWithInfraRed(object):

    def __init__(self, loader_config, crs_name, primary_cache_size, enable_secondary_cache, image_loader=cv2.imread):
        self.__name = loader_config['name']
        self.__tile_root_folder_path_rgb = loader_config['folder_RGB']
        self.__tile_root_folder_path_cir = loader_config['folder_CIR']
        self.__final_tile_size = loader_config['final_tile_size']
        self.__image_reader = image_loader
        self.warnings = []
        primary_cache_size_in_tiles = floor(primary_cache_size / (self.__final_tile_size * self.__final_tile_size * 4))
        self.image_cache = LRUCache(maxsize=primary_cache_size_in_tiles)

        self.__image_requests = 0
        self.__image_cache_misses = 0

        if crs_name == "urn:ogc:def:crs:OGC:1.3:CRS84":  # latitude, longitude
            def lat_long_to_tile_eastings_northings_func(x, y, z=None):
                eastings, northings = lat_long_to_fractional_tile_eastings_and_nothings(y, x)
                return tuple(filter(None, [eastings, northings, z]))

            self.__transform_func = partial(transform, func=lat_long_to_tile_eastings_northings_func)

        elif crs_name == "urn:ogc:def:crs:EPSG::27700":  # eastings, northings
            def eastings_northings_to_fractional_tile(eastings, northings, z=None):
                return tuple(filter(None, [eastings / 1000, northings / 1000, z]))
            self.__transform_func = partial(transform, func=eastings_northings_to_fractional_tile)

        else:
            raise ValueError(f'crs_name="{crs_name}" is unsupported')

    @cachedmethod(operator.attrgetter('image_cache'))
    def retrieve_image(self, image_filename):
        self.__image_cache_misses += 1

        image_bgr_file_name = path.join(self.__tile_root_folder_path_rgb, image_filename)
        image_bgr = self.__image_reader(image_bgr_file_name)
        if image_bgr.shape != (self.__final_tile_size, self.__final_tile_size, 3):
            self.warnings.append(f'Image "{image_bgr_file_name}" is sized {image_bgr.shape}'
                                 f' rather than ({self.__final_tile_size}, {self.__final_tile_size}, 3))')
            image_bgr = cv2.resize(image_bgr, dsize=(self.__final_tile_size, self.__final_tile_size))

        image_cir_file_name = path.join(self.__tile_root_folder_path_cir, image_filename)
        image_cir = self.__image_reader(image_cir_file_name)

        # From blue sky docs (APGB-CIR.pdf): NIR, Red, Green (in the RGB channels), false colour enhanced to maximise
        # information content.
        # Given RGB is loaded as BGR, Ir,R,G loaded as G,R,Ir
        image_ir = image_cir[:, :, 2]

        # upscale ir to bgr resolution
        upscaled_image_ir = cv2.resize(image_ir, (image_bgr.shape[1], image_bgr.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)
        upscaled_image_ir = np.reshape(upscaled_image_ir, (image_bgr.shape[1], image_bgr.shape[0], 1))
        image_bgri = np.concatenate((image_bgr, upscaled_image_ir), axis=2)
        return image_bgri

    @staticmethod
    def build_tile_file_name(eastings, northings):
        tile_code = tile_eastings_and_northings_to_tile_code(eastings, northings)
        # example file path: NS/NS46/NS4360.jpg
        file_name = path.join(tile_code[0:2], tile_code[0:3] + tile_code[4],
                              tile_code + '.jpg')
        return file_name

    def download_image(self, max_tile_e, max_tile_n, min_tile_e, min_tile_n):
        width = (max_tile_e - min_tile_e + 1) * self.__final_tile_size
        height = (max_tile_n - min_tile_n + 1) * self.__final_tile_size

        overall_tile = np.zeros((height, width, 4), np.uint8)
        for northings in range(min_tile_n, max_tile_n + 1):
            for eastings in range(min_tile_e, max_tile_e + 1):
                image_tile_file_name = self.build_tile_file_name(eastings, northings)

                self.__image_requests += 1
                image_tile = self.retrieve_image(image_tile_file_name)

                e_offset = (eastings - min_tile_e) * self.__final_tile_size
                n_offset = (max_tile_n - northings) * self.__final_tile_size

                overall_tile[
                    n_offset:(n_offset + self.__final_tile_size),
                    e_offset:(e_offset + self.__final_tile_size),
                    :] = image_tile
        return overall_tile

    def calculate_tile_geometry(self, geometry):
        tile_geometry = self.__transform_func(geom=geometry)
        return tile_geometry

    def get_image_and_mask(self, tile_geometry, debug_base_file_name=None):

        km2_to_m2 = 1000.0 * 1000.0
        surface_area_m2 = tile_geometry.area * km2_to_m2

        min_tile_e = int(tile_geometry.bounds[0])
        min_tile_n = int(tile_geometry.bounds[1])
        max_tile_e = int(tile_geometry.bounds[2])
        max_tile_n = int(tile_geometry.bounds[3])

        image_bgr = self.download_image(max_tile_e, max_tile_n, min_tile_e, min_tile_n)

        # [a, b, d, e, xoff, yoff]
        # x' = a * x + b * y + xoff
        # y' = d * x + e * y + yoff
        m = [self.__final_tile_size, 0, 0, self.__final_tile_size, -min_tile_e * self.__final_tile_size,
             -min_tile_n * self.__final_tile_size]
        affine_geometry = affine_transform(tile_geometry, m)

        min_x = floor(affine_geometry.bounds[0])
        min_y = floor(affine_geometry.bounds[1])
        max_x = floor(affine_geometry.bounds[2])
        max_y = floor(affine_geometry.bounds[3])

        max_y_vertically_flipped = image_bgr.shape[0] - 1 - min_y
        min_y_vertically_flipped = image_bgr.shape[0] - 1 - max_y

        affine = rasterio.Affine(1, 0, min_x, 0, -1, max_y)

        pixels_within_geometry = geometry_mask([affine_geometry],
                                               (max_y_vertically_flipped - min_y_vertically_flipped + 1,
                                                max_x - min_x + 1),
                                               affine, invert=True)

        image_bgri_cropped = image_bgr[min_y_vertically_flipped:max_y_vertically_flipped + 1, min_x:max_x + 1, :]

        tile_file_name = None

        if debug_base_file_name is not None:
            centre_point = tile_geometry.centroid
            tile_code = tile_eastings_and_northings_to_tile_code(centre_point.x, centre_point.y)
            tile_file_name = debug_base_file_name + '-' + tile_code

        return image_bgri_cropped, pixels_within_geometry, surface_area_m2, tile_file_name

    def report_usage(self):
        # # !!! Changed here. We do not use their image loading using masks
        # print(f'Number of map tile requests: {self.image_requests:,}')
        # print(f'Number of map tile cache hits vs misses: '
        #       f'{self.image_cache_hits:,} vs {self.image_cache_misses:,}')
        print()

    @property
    def name(self):
        return self.__name

    @property
    def image_cache_hits(self):
        return self.__image_requests - self.__image_cache_misses

    @property
    def image_requests(self):
        return self.__image_requests

    @property
    def image_cache_misses(self):
        return self.__image_cache_misses

    @property
    def statistics_report(self):
        return f'{self.image_cache_hits:,} cached, {self.image_cache_misses:,} missed;' \
            f' hit rate {self.image_cache_hits / self.image_requests * 100.0:.1f}%'
