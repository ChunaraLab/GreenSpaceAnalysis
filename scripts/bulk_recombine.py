import argparse
import glob
import os
import re
import sys


def get_args(command_line_arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Recombine multiple results from a set of GeoJSON files into a single output (as if a single GeoJSON has been analysed)")

    parser.add_argument('-of', '--output-folder', default='Z:\\outpile',
                        help="Folder where results are to be written")

    parser.add_argument('-rf', '--results-folder', default='Z:\\results',
                        help="Folder containing results of processing")

    parser.add_argument("-i", "--index", default=None, required=True,
                        help=f"What vegetation index to compute (default: None)")

    parser.add_argument("-wl", "--loader", default=None, required=True,
                        help=f"What tile loader to use (default: None)")

    args = parser.parse_args(command_line_arguments)

    return args


def recombine_summary(args, file_parts, image_source, json_name, metric, num_of_parts):
    total_surface_area = 0.0
    total_vegetation_surface_area = 0.0
    for file_num in range(0, num_of_parts):
        file_part = file_parts[file_num]
        part_file_name = f'{file_part[0]}_{file_num + 1}of{num_of_parts}-{file_part[1]}-{file_part[2]}-summary.txt'

        with open(part_file_name) as pf:
            surface_line = pf.readline()
            vegetation_line = pf.readline()

            surface_area_str = re.match('.+:(.+)m.', surface_line)[1]
            vegetation_area_str = re.match('.+:(.+)m.', vegetation_line)[1]

            surface_area = float(surface_area_str.replace(',', ''))
            vegetation_area = float(vegetation_area_str.replace(',', ''))

            total_surface_area += surface_area
            total_vegetation_surface_area += vegetation_area

    vegetation_percentage = (total_vegetation_surface_area / total_surface_area) * 100.0
    summary_file_name = f'{os.path.basename(json_name)}_{image_source}-{metric}-summary.txt'
    with open(os.path.join(args.output_folder, summary_file_name), 'w') as sf:
        print(f'Total surface area: {total_surface_area:,.2f}m²', file=sf)
        print(f'Total vegetation surface area from {metric}: {total_vegetation_surface_area:,.2f}m²'
              f' ({vegetation_percentage:.1f}%)', file=sf)


def recombine_csv(args, file_parts, image_source, json_name, metric, num_of_parts, file_postfix):

    vegetation_file_name = f'{os.path.basename(json_name)}_{image_source}-{metric}-{file_postfix}.csv'
    with open(os.path.join(args.output_folder, vegetation_file_name), 'w') as vf:

        file_part = file_parts[0]
        first_file_name = f'{file_part[0]}_{1}of{num_of_parts}-{file_part[1]}-{file_part[2]}-{file_postfix}.csv'
        with open(first_file_name) as ff:
            first_line = ff.readline()

        vf.writelines(first_line)

        for file_num in range(0, num_of_parts):
            file_part = file_parts[file_num]
            part_file_name = f'{file_part[0]}_{file_num + 1}of{num_of_parts}-{file_part[1]}-{file_part[2]}' \
                f'-{file_postfix}.csv'

            with open(part_file_name) as pf:
                pf.readline()  # Ignore first line - it'll be the same for all files
                all_lines = pf.readlines()
                vf.writelines(all_lines)


def main(command_line_arguments):
    args = get_args(command_line_arguments)

    os.makedirs(args.output_folder, exist_ok=True)

    csv_file_names = glob.glob(os.path.join(args.results_folder, f'*-{args.loader}-{args.index}-vegetation.csv'))
    if len(csv_file_names) == 0:
        print('No files found to process. Exiting...')
        return 0

    output_filename_regex = r'(.*)_(\d+)of(\d+)-(.*)-(.*)-(.*)'
    regex_match = re.match(output_filename_regex, csv_file_names[0])
    json_name = regex_match[1]
    num_of_parts = int(regex_match[3])
    image_source = regex_match[4]
    metric = regex_match[5]

    file_parts = [None] * num_of_parts
    for csv_file_name in csv_file_names:
        regex_match = re.match(output_filename_regex, csv_file_name)
        file_num = int(regex_match[2])

        file_parts[file_num-1] = (regex_match[1], regex_match[4], regex_match[5])

    recombine_summary(args, file_parts, image_source, json_name, metric, num_of_parts)
    recombine_csv(args, file_parts, image_source, json_name, metric, num_of_parts, 'vegetation')
    recombine_csv(args, file_parts, image_source, json_name, metric, num_of_parts, 'toid2uprn')


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])
