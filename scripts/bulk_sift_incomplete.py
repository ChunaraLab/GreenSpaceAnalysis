import argparse
import glob
import os
import platform
import re
import subprocess
import sys


from green_spaces import analyse_polygons


def get_args(command_line_arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Detect GeoJSON in outpile without matching results file and move back to inpile for reprocessing")

    parser.add_argument('-if', '--inpile-folder', required=True,
                        help="Folder where files to process are stored")

    parser.add_argument('-of', '--outpile-folder', required=True,
                        help="Folder where results are stored")

    parser.add_argument('-rf', '--results-folder', required=True,
                        help="Folder containing results of processing")

    args = parser.parse_args(command_line_arguments)

    return args


def main(command_line_arguments):
    args = get_args(command_line_arguments)

    print('Searching for geojson in outpile without matching results...')

    os.makedirs(args.inpile_folder, exist_ok=True)

    csv_file_names = glob.glob(os.path.join(args.results_folder, f'*-*-*-vegetation.csv'))
    if len(csv_file_names) == 0:
        print('No results files found to process. Exiting...')
        return 0

    output_filename_regex = r'(.*)_(\d+)of(\d+)-(.*)-(.*)-(.*)'
    regex_match = re.match(output_filename_regex, csv_file_names[0])
    json_name = regex_match[1]
    num_of_parts = int(regex_match[3])
    image_source = regex_match[4]
    metric = regex_match[5]

    csv_file_parts = [None] * num_of_parts
    for csv_file_name in csv_file_names:
        regex_match = re.match(output_filename_regex, csv_file_name)
        file_num = int(regex_match[2])
        csv_file_parts[file_num-1] = (regex_match[1], regex_match[4], regex_match[5])

    outpile_geojson_names = glob.glob(os.path.join(args.outpile_folder, f'*.geojson'))
    outpile_filename_regex = r'(.*)_(\d+)of(\d+).geojson'

    for outpile_geojson_name in outpile_geojson_names:
        regex_match = re.match(outpile_filename_regex, outpile_geojson_name)
        file_num = int(regex_match[2])

        if csv_file_parts[file_num - 1] is None:
            print(f'No results for "{outpile_geojson_name}"; moving back to inpile...')
            inpile_file_name = os.path.join(args.inpile_folder, os.path.basename(outpile_geojson_name))
            os.rename(outpile_geojson_name, inpile_file_name)

    print('Searching for geojson in outpile without matching results... done')


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])
