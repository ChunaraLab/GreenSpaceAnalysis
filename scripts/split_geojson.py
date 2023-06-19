import argparse
import json
import sys
from os import path
from math import ceil
from tqdm import tqdm


def get_args(command_line_arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Split GeoJSON files into N features per file (ready for distribution)")

    parser.add_argument("geojson_filename", metavar='<geojson input file name>',
                        help="File name of a GeoJSON file to analyse vegetation coverage")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Report detailed progress and parameters")

    parser.add_argument('-fpf', '--features-per-file', type=int, default=10000,
                        help="Number of features to output per file")

    args = parser.parse_args(command_line_arguments)

    return args


def main(command_line_arguments):
    args = get_args(command_line_arguments)

    base_file_name = path.splitext(args.geojson_filename)[0]

    with open(args.geojson_filename) as geojson_file:
        json_data = json.load(geojson_file)

    json_features = json_data['features']
    num_features = len(json_features)
    num_groups = ceil(num_features / args.features_per_file)

    group_id = 1
    num_in_group = 0

    subjson_data = {
        "type": json_data["type"],
        "name": json_data["name"],
        "crs": json_data["crs"],
        "features": []
    }

    for feature in tqdm(json_features, unit='feature',
                        desc=f'Extracting features into sets of {args.features_per_file}'):
        subjson_data["features"].append(feature)
        num_in_group += 1

        if num_in_group == args.features_per_file:
            with open(f'{base_file_name}_{group_id}of{num_groups}.geojson', 'w') as outfile:
                json.dump(subjson_data, outfile)
            subjson_data["features"] = []
            num_in_group = 0
            group_id += 1

    if num_in_group > 0:
        with open(f'{base_file_name}_{group_id}of{num_groups}.geojson', 'w') as outfile:
            json.dump(subjson_data, outfile)


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])
