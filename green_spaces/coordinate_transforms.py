import math

from pyproj import Proj, transform

proj_WGS84 = Proj('+init=EPSG:4326')
proj_OSGB36 = Proj('+init=EPSG:27700')


# Code taken from https://gist.github.com/sebastianleonte/617628973f88792cd097941220110233

# Reference https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames


def lat_long_to_web_mercator_tile_yx(zoom, latitude, longitude):
    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    num_tiles = 1 << zoom
    # Find the x_point given the longitude
    point_x = (0.5 + longitude / 360.0) * num_tiles
    # Convert the latitude to radians and take the sine
    sin_y = math.sin(latitude * (math.pi / 180.0))
    # Calculate the y coordinate
    point_y = (0.5 + math.log((1 + sin_y) / (1 - sin_y)) / (-4 * math.pi)) * num_tiles
    return point_y, point_x


def lat_long_to_web_mercator_tile_int_yx(zoom, latitude, longitude):
    point_y, point_x = lat_long_to_web_mercator_tile_yx(zoom, latitude, longitude)
    return int(point_y), int(point_x)


# from https://cdn.rawgit.com/chrisveness/geodesy/v1.1.3/osgridref.js
def tile_eastings_and_northings_to_tile_code(eastings, northings):
    # get the 100km-grid indices
    e100km = math.floor(eastings / 100)
    n100km = math.floor(northings / 100)

    if e100km < 0 or e100km > 7 or n100km < 0 or n100km > 12:
        return None

    # translate those into numeric equivalents of the grid letters
    letter_1_code = (19 - n100km) - (19 - n100km) % 5 + math.floor((e100km + 10) / 5)
    letter_2_code = (19 - n100km) * 5 % 25 + e100km % 5

    # compensate for skipped 'I' and build grid letter-pairs
    if letter_1_code > 7:
        letter_1_code += 1

    if letter_2_code > 7:
        letter_2_code += 1

    letter1 = chr(ord('A') + letter_1_code)
    letter2 = chr(ord('A') + letter_2_code)

    tile_digits_eastings = int(eastings % 100)
    tile_digits_northings = int(northings % 100)
    tile_code = f'{letter1}{letter2}{tile_digits_eastings:02}{tile_digits_northings:02}'
    return tile_code


def tile_code_to_tile_eastings_and_northings(tile_code):
    if len(tile_code) != 6:
        raise ValueError('tile_code must be 6 characters')

    letter1 = tile_code[0]
    letter2 = tile_code[1]
    tile_digits_eastings = int(tile_code[2:4])
    tile_digits_northings = int(tile_code[4:6])

    letter_1_code = ord(letter1) - ord('A')
    letter_2_code = ord(letter2) - ord('A')

    if letter_1_code > 8:
        letter_1_code -= 1

    if letter_2_code > 8:
        letter_2_code -= 1

    # convert grid letters into 100km-square indexes from false origin (grid square SV):
    e100km = ((letter_1_code - 2) % 5) * 5 + (letter_2_code % 5)
    n100km = (19 - math.floor(letter_1_code / 5) * 5) - math.floor(letter_2_code / 5)

    if e100km < 0 or e100km > 7 or n100km < 0 or n100km > 12:
        return None, None

    eastings = e100km * 100 + tile_digits_eastings
    northings = n100km * 100 + tile_digits_northings

    return eastings, northings


# Ref https://gis.stackexchange.com/questions/295183/geopandas-fails-to-transform-coordinates-from-osgb36-to-wgs84
def lat_long_to_national_grid_tile_pixel_xy(latitude, longitude, image_size_in_pixels):
    eastings, northings = transform(proj_WGS84, proj_OSGB36, longitude, latitude)
    tile_code = tile_eastings_and_northings_to_tile_code(eastings // 1000, northings // 1000)
    fraction_x = (eastings % 1000) / 1000
    fraction_y = (northings % 1000) / 1000
    pixel_x = int(image_size_in_pixels * fraction_x)
    pixel_y = int((image_size_in_pixels - 1) - (image_size_in_pixels * fraction_y))

    return tile_code, pixel_x, pixel_y


# Ref https://gis.stackexchange.com/questions/295183/geopandas-fails-to-transform-coordinates-from-osgb36-to-wgs84
def lat_long_to_fractional_tile_eastings_and_nothings(latitude, longitude):
    eastings, northings = transform(proj_WGS84, proj_OSGB36, longitude, latitude)
    return eastings / 1000, northings / 1000
