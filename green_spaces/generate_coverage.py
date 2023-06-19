# [A-Z][A-Z]/AB[00-99]/AB9[0-9]9[0-9]/.jpeg
# ST/ST00/ST0102
import argparse
import json
import os
import sys
import humanfriendly
import xmltodict
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib import cm

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from green_spaces.coordinate_transforms import tile_code_to_tile_eastings_and_northings


def folders_in_folder(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def files_in_folder(dir, extension):
    names = []
    for name in os.listdir(dir):
        if os.path.splitext(name)[1] == extension:
            if os.path.isfile(os.path.join(dir, name)):
                names.append(name)
    return names


class Coverage(object):
    def __init__(self, tile_size):
        self.tile_size = tile_size
        self.white_tile = (np.ones(shape=(tile_size, tile_size, 3)) * 255).astype(np.uint8)

    @property
    def data_dims(self):
        return 3

    @property
    def data_type(self):
        return np.uint8

    @property
    def file_ext(self):
        return '.jpg'

    @property
    def report_file_postfix(self):
        return '-coverageOnly'

    def process_tile(self, file_name):
        return self.white_tile

    def output_summary_data(self, summary_data, file_name_base):
        cv2.imwrite(file_name_base + '.png', summary_data)

    def report_state(self, f):
        pass


class Thumbnail(object):
    def __init__(self, tile_size):
        self.tile_size = tile_size
        self.corrupted_tiles = []
        self.number_of_pixels_processed = 0

    @property
    def data_dims(self):
        return 3

    @property
    def data_type(self):
        return np.uint8

    @property
    def file_ext(self):
        return '.jpg'

    @property
    def report_file_postfix(self):
        return ''

    def process_tile(self, full_file_name):
        try:
            img = cv2.imread(full_file_name)
        except:
            img = None

        if img is None:
            self.corrupted_tiles.append(full_file_name)
            return None

        self.number_of_pixels_processed += img.shape[0] * img.shape[1]

        thumbnail = cv2.resize(img, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA)
        return thumbnail

    def output_summary_data(self, summary_data, file_name_base):
        cv2.imwrite(file_name_base + '.png', summary_data)

    def report_state(self, f):
        print(f'Number of pixels processed: {humanfriendly.format_size(self.number_of_pixels_processed)} '
              f'pixels ({self.number_of_pixels_processed:,})', file=f)
        print(f'Corrupted tiles: {self.corrupted_tiles}', file=f)


class Flights(object):
    def __init__(self, tile_size):
        self.tile_size = tile_size
        self.date_text_to_datetime = {}
        self.corrupted_tiles = []
        self.datetime_corrupted = []
        self.number_of_pixels_processed = 0
        self.max_date_float = 0
        self.min_date_float = 99999999
        self.min_time_of_year = 1
        self.max_time_of_year = 0

    @property
    def data_dims(self):
        return 1

    @property
    def data_type(self):
        return np.float

    @property
    def file_ext(self):
        return '.xml'

    @property
    def report_file_postfix(self):
        return '-flights'

    def process_tile(self, full_file_name):

        with open(full_file_name, 'rb') as f:
            json = xmltodict.parse(f, )
            if json is None:
                self.corrupted_tiles.append(full_file_name)
                return None

            root_key = next(iter(json))
            date_flown_text = json[root_key]['osgb:dateFlown']

            if isinstance(date_flown_text, list):
                date_flown_text = date_flown_text[0]

            if date_flown_text not in self.date_text_to_datetime:

                try:
                    date_flown_datetime = datetime.strptime(date_flown_text, '%Y-%m-%d')  # 2010-05-18
                except ValueError:
                    try:
                        date_flown_datetime = datetime.strptime(date_flown_text, '%d/%m/%Y')  # 16/07/2006
                    except ValueError:
                        self.datetime_corrupted.append(full_file_name)
                        return None

                date_flown_iso = date_flown_datetime.isocalendar()
                date_flown_float = date_flown_iso[0] + (date_flown_iso[1] * 7 + date_flown_iso[2]) / 366.0
                self.date_text_to_datetime[date_flown_text] = (date_flown_float, date_flown_datetime)

                if date_flown_float > self.max_date_float:
                    self.max_date_float = date_flown_float
                if date_flown_float < self.min_date_float:
                    self.min_date_float = date_flown_float
            else:
                date_flown_float = self.date_text_to_datetime[date_flown_text][0]

            data = np.ones(shape=(self.tile_size, self.tile_size, 1), dtype=self.data_type) * date_flown_float

            return data

    def output_summary_data(self, summary_data, file_name_base):
        map_title_base = os.path.basename(file_name_base)

        colour_map = cm.get_cmap(name='viridis')

        if self.max_date_float == self.min_date_float:
            normalised_summary_data = np.zeros(shape=summary_data.shape)
        else:
            colour_map_scale = 1.0 / (self.max_date_float - self.min_date_float)
            normalised_summary_data = (summary_data - self.min_date_float) * colour_map_scale

        max_date_datetime = datetime(int(self.max_date_float), 1, 1) \
            + timedelta((self.max_date_float - int(self.max_date_float)) * 365)
        min_date_datetime = datetime(int(self.min_date_float), 1, 1) \
            + timedelta((self.min_date_float - int(self.min_date_float)) * 365)

        background_black_summary_rgb = self.convert_to_bitmap_and_add_key(
            colour_map,
            f'{map_title_base} Capture Date', '',
            max_date_datetime.strftime("%B %d %Y"),
            min_date_datetime.strftime("%B %d %Y"),
            normalised_summary_data, summary_data)

        cv2.imwrite(file_name_base + '-captureDate.png', background_black_summary_rgb)

        time_of_year_summary_data = summary_data - np.floor(summary_data)

        self.min_time_of_year = np.where(summary_data == 0, 1, time_of_year_summary_data).min()
        self.max_time_of_year = time_of_year_summary_data.max()

        max_time_of_year_datetime = datetime(2000, 1, 1) + timedelta(self.max_time_of_year * 365)
        min_time_of_year_datetime = datetime(2000, 1, 1) + timedelta(self.min_time_of_year * 365)

        background_black_summary_rgb = self.convert_to_bitmap_and_add_key(
            colour_map,
            f'{map_title_base}',
            f'Time of Year (from {min_time_of_year_datetime.strftime("%d %B")}'
            f' to {max_time_of_year_datetime.strftime("%d %B")})',
            '31st December',
            '1st January',
            time_of_year_summary_data, summary_data)

        cv2.imwrite(file_name_base + '-captureTimeOfYear.png', background_black_summary_rgb)

    def convert_to_bitmap_and_add_key(self, colour_map, title, subtitle, max_label, min_label,
                                      normalised_summary_data_source, summary_data_source):

        normalised_summary_data = normalised_summary_data_source.copy()
        summary_data = summary_data_source.copy()

        for x in range(normalised_summary_data.shape[1] - (16 * self.tile_size), normalised_summary_data.shape[1]):
            for y in range(0, 256 * self.tile_size):
                normalised_summary_data[y + (4 * self.tile_size), x - (4 * self.tile_size), 0] = 1.0 - (
                            y / (255.0 * self.tile_size))
                summary_data[y + (4 * self.tile_size), x - (4 * self.tile_size), 0] = -1

        # https://matplotlib.org/gallery/color/colormap_reference.html
        summary_image_rgba = colour_map(normalised_summary_data[:, :, 0], bytes=True)
        background_black_summary_rgb = np.where(summary_data == 0, 0, summary_image_rgba[:, :, 0:3])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        font_colour = (255, 255, 255)
        line_type = 2

        bottom_left_corner_of_text = ((16 * self.tile_size), (16 * self.tile_size))
        cv2.putText(background_black_summary_rgb, title,
                    bottom_left_corner_of_text, font, font_scale*1.2, font_colour, line_type)

        bottom_left_corner_of_text = ((16 * self.tile_size), (16 * self.tile_size) + (font_scale * 4 * self.tile_size))
        cv2.putText(background_black_summary_rgb, subtitle,
                    bottom_left_corner_of_text, font, font_scale*1.2, font_colour, line_type)

        bottom_left_corner_of_text = (
            normalised_summary_data.shape[1] - (16 * self.tile_size)
            - (11 * self.tile_size * len(max_label)),
            (16 * self.tile_size))
        cv2.putText(background_black_summary_rgb, max_label,
                    bottom_left_corner_of_text, font, font_scale, font_colour, line_type)

        bottom_left_corner_of_text = (
            normalised_summary_data.shape[1] - (16 * self.tile_size) - (11 * self.tile_size * len(min_label)),
            (16 * self.tile_size) + 256 * self.tile_size - 12 * self.tile_size)
        cv2.putText(background_black_summary_rgb, min_label,
                    bottom_left_corner_of_text, font, font_scale, font_colour, line_type)

        return background_black_summary_rgb

    def report_state(self, f):
        if self.min_time_of_year > self.max_time_of_year:
            raise ValueError('self.min_time_of_year > self.ma_time_of-year;'
                             ' have you run output_summary_data() before calling report_state()?')

        print(f'Number of pixels processed: {humanfriendly.format_size(self.number_of_pixels_processed)} '
              f'pixels ({self.number_of_pixels_processed:,})', file=f)
        print(f'Date time corrupted tiles: {self.datetime_corrupted}', file=f)
        print(f'Corrupted tiles: {self.corrupted_tiles}', file=f)
        print(f'Maximum date: {self.max_date_float}', file=f)
        print(f'Minimum date: {self.min_date_float}', file=f)

        max_time_of_year_datetime = datetime(2000, 1, 1) + timedelta(self.max_time_of_year * 365)
        min_time_of_year_datetime = datetime(2000, 1, 1) + timedelta(self.min_time_of_year * 365)

        print(f'Maximum time of year: {self.max_time_of_year} ({max_time_of_year_datetime.strftime("%d %B")})', file=f)
        print(f'Minimum time of year: {self.min_time_of_year} ({min_time_of_year_datetime.strftime("%d %B")})', file=f)


def produce_overall_map(root_data_folder, data_type, tile_processor, tile_size=8, use_tqdm=True, root_folder=''):
    # letters = [chr(ord('A') + letter) for letter in range(26) if letter != ord('I') - ord('A')]

    # UK is covered by 2x3 squares, each 500 x 500km:
    #  HJ
    #  NO
    #  ST
    #
    # SV to JG
    #
    # Each 500 x 500km divided into 5x5 letters, forming 100x100km two letter squares
    # Each square has 100 x 100 files... each 1knm square
    #
    # Hence 500 x 500 files (tiles) per initial letter
    # summary_data_shape = [3 * 500 * tile_size, 2 * 500 * tile_size, tile_processor.data_dims]
    #
    # Now reduce to 7 * 13 100km squares
    summary_data_shape = [13 * 100 * tile_size, 7 * 100 * tile_size, tile_processor.data_dims]
    print(f'Summary data shape: {summary_data_shape[0]:,} x {summary_data_shape[1]:,} pixels')

    summary_data = np.zeros(shape=summary_data_shape, dtype=tile_processor.data_type)

    origin_x, origin_y = tile_code_to_tile_eastings_and_northings('SV0000')

    eastings_northings_failures = []
    odd_file_names = []

    if use_tqdm:
        iter_wrapper = tqdm
    else:
        def iter_wrapper(iterator, desc, ascii):
            # return [iterator[0]]
            return iterator

    print()

    number_of_1km_tiles_processed = 0

    # if False:
    # for letter_pair in ['SV']:
    # for letter_pair in ['ST']:
    for letter_pair in iter_wrapper(folders_in_folder(root_data_folder), desc='100km tiles', ascii=True):
        if not use_tqdm:
            print(letter_pair + ': ', end='')

        current_folder = os.path.join(root_data_folder, letter_pair)

        for letter_number_pair in iter_wrapper(folders_in_folder(current_folder), desc='10km tiles in ' + letter_pair,
                                               ascii=True):
            if not use_tqdm:
                print(letter_number_pair + ', ', end='')
            inner_folder = os.path.join(current_folder, letter_number_pair)

            for file_name in iter_wrapper(files_in_folder(inner_folder, tile_processor.file_ext),
                                          desc='1km tiles in ' + letter_number_pair, ascii=True):

                file_name_base = os.path.splitext(file_name)[0]
                if len(file_name_base) != 6:
                    odd_file_names.append(file_name)
                    continue

                number_of_1km_tiles_processed += 1

                full_file_name = os.path.join(inner_folder, file_name)

                thumbnail = tile_processor.process_tile(full_file_name)

                if thumbnail is None:
                    continue

                tile_code = os.path.splitext(file_name)[0]
                x, y = tile_code_to_tile_eastings_and_northings(tile_code)

                if x is None:
                    eastings_northings_failures.append(tile_code)
                    continue

                x -= origin_x
                y -= origin_y

                x *= tile_size
                y *= tile_size

                y = summary_data.shape[0] - y - tile_size

                summary_data[y:y + tile_size, x:x + tile_size, :] = thumbnail

        if not use_tqdm:
            print()
            print()

    os.makedirs('output', exist_ok=True)
    file_postfix = tile_processor.report_file_postfix

    tile_processor.output_summary_data(summary_data, os.path.join('output', f'{data_type}-thumbnail{file_postfix}'))

    with open(os.path.join('output', f'{data_type}-status{file_postfix}.txt'), 'w') as f:
        tile_processor.report_state(f)
        print(f'Number of 1km tiles processed: {number_of_1km_tiles_processed:,}', file=f)
        print(f'Skipped files due to odd file names: {odd_file_names}', file=f)
        print(f'Skipped tiles due to failing to map to eastings-northings : {eastings_northings_failures}', file=f)

    # to move cursor past tqdm output...
    print()
    print()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(command_line_arguments=sys.argv[1:]):
    this_file_name = os.path.realpath(__file__)
    json_file_name = os.path.join(os.path.split(this_file_name)[0], 'analyse_polygons.json')
    with open(json_file_name, 'r') as json_file:
        config = json.load(json_file)

    parser = argparse.ArgumentParser(description="Generate overall map from OSGB folder hierarchy")

    dataset_names = []
    if 'loaders' in config:
        for loader in config['loaders']:
            if loader["class"] == "green_spaces.image_loaders.OrdnanceSurveyMapLoader":
                dataset_names.append(loader['name'])

    parser.add_argument('dataset', choices=dataset_names,
                        help="Which dataset to analyse")

    parser.add_argument('-ts', '--tile-size', type=int, default=8, help='Tile size each image is mapped to')

    parser.add_argument('-tqdm', '--use-tqdm', type=str2bool, default=True,
                        help='Use TQDM to display completion graphs')

    parser.add_argument('-ca', '--coverage-analysis', choices=['thumbnail', 'coverage', 'flights'], default='thumbnail',
                        help='Data represented in summary image')

    parser.add_argument('-rf', '--root-folder', default='H:\\',
                        help='Root folder where aerial photography is stored')

    args = parser.parse_args(command_line_arguments)

    if args.coverage_analysis == 'thumbnail':
        tile_processor = Thumbnail(args.tile_size)
    elif args.coverage_analysis == 'coverage':
        tile_processor = Coverage(args.tile_size)
    elif args.coverage_analysis == 'flights':
        tile_processor = Flights(args.tile_size)
    else:
        raise ValueError(f'Unknown coverage analysis type: {args.coverage_analysis}')

    dataset_path = None
    if 'loaders' in config:
        for loader in config['loaders']:
            if loader["class"] == "green_spaces.image_loaders.OrdnanceSurveyMapLoader" and loader['name'] == args.dataset:
                dataset_path = loader['folder']

    produce_overall_map(dataset_path, args.dataset, tile_processor, args.tile_size, args.use_tqdm, args.root_folder)


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])
