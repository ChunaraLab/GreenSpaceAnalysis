import argparse
import importlib
import json
import random
import sys
from os import path, makedirs
import numpy as np

import humanfriendly

from green_spaces.calculate_indices import calculate_feature_indices


def class_from_name(full_class_name):
    loader_module_name, loader_class_name = full_class_name.rsplit(".", 1)
    loader_module = importlib.import_module(loader_module_name)
    loader_class = getattr(loader_module, loader_class_name)
    return loader_class


def get_args(command_line_arguments=sys.argv[1:]):
    json_file_name = path.splitext(path.realpath(__file__))[0] + '.json'
    with open(json_file_name, 'r') as json_file:
        config = json.load(json_file)

    parser = argparse.ArgumentParser(description="Parse GeoJSON files, download imagery covered by GeoJSON"
                                                 " and calculate requested image metrics within each GeoJSON polygon")

    parser.add_argument("geojson_filename", metavar='<geojson input file name>',
                        help="File name of a GeoJSON file to analyse vegetation coverage")

    parser.add_argument("-o", "--output-folder", default="output",
                        help="Folder name where results of vegetation coverage are output")

    parser.add_argument('-pc', '--primary-cache-size', default='0',
                        help="Memory to allocate for map tiles primary cache (0=no primary cache);"
                             " uses human friendly format e.g. 12M=12,000,000")

    parser.add_argument('-esc', '--enable-secondary-cache', action='store_true',
                        help="Use local storage to hold copies of all downloaded data and avoid multiple downloads")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Report detailed progress and parameters")

    parser.add_argument('-fng', '--first-n-gardens', type=int, default=0,
                        help="Only process first N gardens")

    parser.add_argument('-rng', '--random-n-gardens', type=int, default=0,
                        help="Process random N gardens")

    parser.add_argument('-opv', '--only-paint-vegetation', action='store_true',
                        help='Only paint vegetation pixels in output bitmaps')

    loader_names = []
    if 'loaders' in config:
        for loader in config['loaders']:
            loader_names.append(loader['name'])

    parser.add_argument("-wl", "--loader", default=None, choices=loader_names,
                        help=f"What tile loader to use (default: None)")

    index_names = []
    index_help = None
    if 'indices' in config:
        for index in config['indices']:
            # index_class = class_from_name(index['class'])
            index_names.append(index['short_name'])
            # index['name'] = index['short_name']

            help_description = f"'{index['short_name']}' ({index['full_name']})"
            if index_help is None:
                index_help = help_description
            else:
                index_help += ', ' + help_description

    parser.add_argument("-i", "--index", default=None, choices=index_names, nargs='+',
                        help=f"What vegetation index to compute (default: None); options are: {index_help}")

    parser.add_argument('-di', '--downsampled-images', default=0, type=int, choices=[0, 1, 2, 4],
                        help="Dump downsampled images for each garden for debugging/verification "
                             "('0' does not produce images, '1' produces unscaled images, "
                             "'2' produces 1:2 downsampled images, '4' produces 1:4 downsampled images")

    args = parser.parse_args(command_line_arguments)

    if args.loader is None:
        print('No loader selected; select define one with --loader')
        print()
        parser.print_usage()
        exit(1)

    if args.index is None:
        print('No vegetation index selected; select define one with --index')
        print()
        parser.print_usage()
        exit(1)

    if args.first_n_gardens != 0 and args.random_n_gardens != 0:
        print('Cannot request both random N gardens and first N gardens')
        print()
        parser.print_usage()
        exit(1)

    args.primary_cache_size = humanfriendly.parse_size(args.primary_cache_size)

    if args.verbose:
        print(f'Using tile loader: {args.loader}')
        print(f'Loading GeoJSON: {args.geojson_filename}')
        print(f'Processing with index: {args.index}')
        print(f'Process first N gardens: {args.first_n_gardens}')
        print(f'Process random N gardens: {args.random_n_gardens}')
        print(f'Producing downsampled images: {args.downsampled_images}')
        print(f'In images, only paint vegetation: {args.only_paint_vegetation}')
        print(f'Primary cache size: {humanfriendly.format_size(args.primary_cache_size)}')
        print('Secondary local storage cache: ' + ('Enabled' if args.enable_secondary_cache else 'Disabled'))

    args.loader_config = [loader for loader in config['loaders'] if loader['name'] == args.loader][0]
    args.indices_config = [index for index in config['indices'] if index['short_name'] in args.index]

    return args


def report_feature_analysis(feature_indices, vegetation_indices, map_loader_name, base_output_file_name, crs_name,
                            warnings):

    output_file_name = base_output_file_name + map_loader_name

    for vegetation_index in vegetation_indices:
        output_file_name += '-' + vegetation_index.short_name

    if len(warnings) > 0:
        with open(output_file_name + '-warnings.txt', 'w') as warnings_file:
            for warning in warnings:
                print(warning, file=warnings_file)

    with open(output_file_name + '-summary.txt', 'w') as summary_file:

        total_surface_area_m2 = 0
        total_vegetation_area_m2 = [0] * len(vegetation_indices)
        for feature_index in feature_indices:
            feature_id, feature_uprn, garden_centroid, surface_area_m2, vegetation_results = feature_index
            total_surface_area_m2 += surface_area_m2

            for index, vegetation_result in enumerate(vegetation_results):
                (fraction_of_vegetation_present, num_pixels_in_polygon, vegetation_image) = vegetation_result
                vegetation_area_m2 = surface_area_m2 * fraction_of_vegetation_present
                total_vegetation_area_m2[index] += vegetation_area_m2

        print(f'Total surface area: {total_surface_area_m2:,.2f}m²', file=summary_file)

        for index, vegetation_index in enumerate(vegetation_indices):
            print(f'Total vegetation surface area from {vegetation_index.short_name}:'
                  f' {total_vegetation_area_m2[index]:,.2f}m²'
                  f' ({100*total_vegetation_area_m2[index]/total_surface_area_m2:0.1f}%)', file=summary_file)

        print(f'Garden centroid output with co-ordinate reference system [{crs_name}]', file=summary_file)

    with open(output_file_name + '-vegetation.csv', 'w') as vegetation_results_file:
        with open(output_file_name + '-toid2uprn.csv', 'w') as toid2uprn_file:
            print(f'feature id, garden centroid x, garden centroid y, surface area m²', file=vegetation_results_file, end='')
            for vegetation_index in vegetation_indices:
                print(f', fraction classed as vegetation by {vegetation_index.short_name}', file=vegetation_results_file, end='')
            print(file=vegetation_results_file)

            print(f'feature id, feature uprn', file=toid2uprn_file)

            for feature_index in feature_indices:
                feature_id, feature_uprns, garden_centroid, surface_area_m2, vegetation_results = feature_index
                print(f'{feature_id}, {garden_centroid[0]}, {garden_centroid[1]}, {surface_area_m2}', file=vegetation_results_file, end='')

                for vegetation_result in vegetation_results:
                    (fraction_of_vegetation_present, num_pixels_in_polygon, vegetation_image) = vegetation_result
                    print(f', {fraction_of_vegetation_present}', file=vegetation_results_file, end='')
                print(file=vegetation_results_file)

                if type(feature_uprns) is list:
                    for feature_uprn in feature_uprns:
                        print(f'{feature_id}, {feature_uprn}', file=toid2uprn_file)
                elif type(feature_uprns) is str:
                    uprn_list = feature_uprns.replace('{','').replace('}','').split(',')
                    for feature_uprn in uprn_list:
                        print(f'{feature_id}, {feature_uprn}', file=toid2uprn_file)
                elif feature_uprns is None:
                    # nothing to record
                    pass
                else:
                    raise ValueError(f"'feature_uprns' is of unhandled type {type(feature_uprns)}")


def main(command_line_arguments):
    args = get_args(command_line_arguments)

    random.seed(42)

    with open(args.geojson_filename) as geojson_file:
        json_data = json.load(geojson_file)

    crs_name = "urn:ogc:def:crs:OGC:1.3:CRS84"  # According to standard, default is lat-long (CRS84 or WGS84)
    if "crs" in json_data:
        if "properties" in json_data["crs"]:
            if "name" in json_data["crs"]["properties"]:
                crs_name = json_data["crs"]["properties"]["name"]

    loader_class = class_from_name(args.loader_config['class'])
    # print("******************")
    # print(loader_class)
    map_loader = loader_class(args.loader_config, crs_name, args.primary_cache_size, args.enable_secondary_cache)

    vegetation_indices = [class_from_name(index_config['class'])(index_config) for index_config in args.indices_config]

    # print("******************")
    # print(vegetation_indices)

    for vegetation_index in vegetation_indices:
        if args.loader_config['format'] != vegetation_index.input_format():
            print()
            print(f"Error: '{args.loader_config['name']}' loader produces '{args.loader_config['format']}' "
                  f"but vegetation index '{vegetation_index.short_name}' expects '{vegetation_index.input_format()}'")
            exit(1)

    output_folder_name = path.join(args.output_folder,
                                   f'{args.loader}'
                                   + (f'-first{args.first_n_gardens}' if args.first_n_gardens > 0 else '')
                                   + (f'-rand{args.random_n_gardens}' if args.random_n_gardens > 0 else '')
                                   )

    json_features = json_data['features']

    if args.first_n_gardens != 0:
        json_features = json_features[:args.first_n_gardens]
    elif args.random_n_gardens != 0:
        json_features = random.sample(json_features, args.random_n_gardens)


    makedirs(output_folder_name, exist_ok=True)

    # print("********* map_loader is: *********")
    # print(map_loader)

    # print("********* json_features is: *********")
    # print(json_features)

    # print("********* vegetation_indices is: *********")
    # print(vegetation_indices)

    # print("********* args.downsampled_images is: *********")
    # print(args.downsampled_images)

    # print("********* args.only_paint_vegetation is: *********")
    # print(args.only_paint_vegetation)


    feature_indices, vegetation_presence = calculate_feature_indices(map_loader, json_features, vegetation_indices, output_folder_name,
                                                args.downsampled_images, args.only_paint_vegetation)

    map_loader.report_usage()

    base_output_file_name = path.join(output_folder_name, path.basename(path.splitext(args.geojson_filename)[0]) + '-')
    report_feature_analysis(feature_indices, vegetation_indices, map_loader.name, base_output_file_name, crs_name,
                            map_loader.warnings)

    # print("***** output folder name *****")
    # print(output_folder_name)
    np.save(output_folder_name+'/vegetation_index_'+'HP0102.npy', np.array(vegetation_presence))

    print()
    print()


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])

# test runs
# old code: analyse_polygons.py --osgb36-loader "25cm RGB aerial"
# --index hsv data\22052018_cardiff_residential_gardens.geojson
# (83,178 cached, 78 missed; hit rate 99.9%): 100%|██████████| 79643/79643 [2:46:10<00:00,  7.64feature/s]
#
# new code 19 Nov 2018: analyse_polygons.py --osgb36-loader "25cm RGB aerial"
#  --index hsv data\22052018_cardiff_residential_gardens.geojson
# (83,178 cached, 78 missed; hit rate 99.9%): 100%|██████████| 79643/79643 [1:07:59<00:00, 19.52feature/s]
