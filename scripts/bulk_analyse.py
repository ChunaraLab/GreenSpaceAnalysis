import argparse
import glob
import os
import platform
import subprocess
import sys


def get_args(command_line_arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Process a folder full of GeoJSON files")

    parser.add_argument('-if', '--inpile-folder', default='Z:\\inpile',
                        help="Folder containing geojson files to be processed")

    parser.add_argument('-of', '--outpile-folder', default='Z:\\outpile',
                        help="Folder containing processed geojson files")

    parser.add_argument('-rf', '--results-folder', default='Z:\\results',
                        help="Folder to contain results of processing")

    parser.add_argument('-pf', '--processing-folder', default='Z:\\processing',
                        help="Folder to contain geojson files whilst they are processed")

    parser.add_argument('-pcs', '--primary-cache-size', default='0',
                        help="Memory to allocate for map tiles primary cache (0=no primary cache);"
                             " uses human friendly format e.g. 12M=12,000,000")

    parser.add_argument("-i", "--index", default=None,
                        help=f"What vegetation index to compute (default: None)")

    parser.add_argument("-wl", "--loader", default=None,
                        help=f"What tile loader to use (default: None)")

    args = parser.parse_args(command_line_arguments)

    return args


def main(command_line_arguments):
    args = get_args(command_line_arguments)

    os.makedirs(args.outpile_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.processing_folder, exist_ok=True)

    machine_process_folder_name = platform.node() + '_P' + str(os.getpid())
    processing_folder = os.path.join(args.processing_folder, machine_process_folder_name)
    print(f'Creating processing folder "{processing_folder}"')
    os.makedirs(processing_folder, exist_ok=False)

    while True:
        # 1. Scan 'inpile' folder; if no files, exit
        geojson_files = glob.glob(os.path.join(args.inpile_folder, '*.geojson'))
        if len(geojson_files) == 0:
            print('No files found to process. Exiting...')
            return 0

        # 2. move a file to 'processing/machinename-processid'
        geojson_file_name = geojson_files[0]

        base_geojson_file_name = os.path.basename(geojson_file_name)
        target_geojson_file_name = os.path.join(args.processing_folder, machine_process_folder_name,
                                                base_geojson_file_name)
        try:
            # from https://docs.python.org/3/library/os.html
            # "...If successful, the renaming will be an atomic operation (this is a POSIX requirement)..."
            os.rename(geojson_file_name, target_geojson_file_name)

        #    on failure, go to 1
        except OSError:
            print(f'Failed to allocate "{geojson_file_name}", trying another...')
            continue

        # 3. process file in 'processing/machinename-processid'
        try:
            print(f'Processing "{geojson_file_name}"...')
            # analyse_polygons.main([...
            arguments = [
                'python', os.path.join('emeraldenv', 'analyse_polygons.py'),

                # 4. output results in 'results' folder
                '--output-folder', args.results_folder,

                '--index', args.index,
                '--primary-cache-size', args.primary_cache_size,
                '--loader', args.loader,
                '--verbose',
                # '-fng', '10',
                target_geojson_file_name
            ]
            print(f'Running: {arguments}')
            subprocess.run(arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(f'Processing "{geojson_file_name}"... complete')
            print()

        except subprocess.CalledProcessError as e:
            print('ERROR Failed to execute')

            inpile_file_name = os.path.join(args.inpile_folder, base_geojson_file_name)
            print(f'geojson file moved from "{target_geojson_file_name}" back to "{inpile_file_name}"')
            os.rename(target_geojson_file_name, inpile_file_name)

            print(f'Removing processing folder "{processing_folder}"')
            os.rmdir(processing_folder)

            error_file_name = os.path.join(args.outpile_folder, '__ERROR-' + machine_process_folder_name
                                           + '-' + os.path.splitext(base_geojson_file_name)[0]
                                           + '.txt')
            print(f'Reporting error to {error_file_name}')
            with open(error_file_name, 'w') as f:
                print(f'Command = "{e.cmd}"', file=f)
                print(file=f)

                stdout = ''
                if e.output is not None:
                     stdout = e.output.decode('ascii')
                print('==================== stdout ==================', file=f)
                print(stdout, file=f)
                print(file=f)

                stderr = ''
                if e.stderr is not None:
                     stderr = e.stderr.decode('ascii')
                print('==================== stderr ==================', file=f)
                print(stderr, file=f)

            print('Exiting...')
            exit(-1)

        # 5. move file from 'processing/machinename-processid' to 'outpile'
        outpile_file_name = os.path.join(args.outpile_folder, base_geojson_file_name)
        os.rename(target_geojson_file_name, outpile_file_name)

    print(f'Removing processing folder "{processing_folder}"')
    os.rmdir(processing_folder)


if __name__ == '__main__':
    main(command_line_arguments=sys.argv[1:])
