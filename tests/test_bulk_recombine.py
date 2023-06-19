import os

import pytest
from pyfakefs.fake_filesystem_unittest import TestCase

from scripts import bulk_recombine


class TestBulkRecombine(TestCase):

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys

    def setup_method(self, method):
        self.setUpPyfakefs()
        self.output_folder = '/output/'
        self.results_folder = '/results/'
        os.makedirs(self.output_folder)
        os.makedirs(self.results_folder)

    def create_text_file(self, file_name, lines):
        with open(file_name, 'w') as f:
            for line in lines:
                print(line, file=f)

    def test_recombine_detects_no_files(self):
        bulk_recombine.main(['--output-folder', self.output_folder,
                             '--results-folder', self.results_folder,
                             '--index', 'metric',
                             '--loader', 'special data'])
        captured = self.capsys.readouterr()

        self.assertEqual('No files found to process. Exiting...\n', captured.out)

    def test_recombine_handles_single_file(self):
        metric_name = 'metric'
        surface_area_m2_1 = 12345.12
        vegetation_surface_area_m2_1 = 4321.93
        vegetation_percentage = vegetation_surface_area_m2_1 / surface_area_m2_1 * 100.0
        expected_summary = [
            f'Total surface area: {surface_area_m2_1:,.2f}m²',
            f'Total vegetation surface area from {metric_name}: {vegetation_surface_area_m2_1:,.2f}m²'
            f' ({vegetation_percentage:.1f}%)'
        ]

        self.create_text_file(
            self.results_folder + f'jsonname_1of1-imageSource-{metric_name}-summary.txt',
            [expected_summary[0], expected_summary[1], 'Garden centroid...'])

        expected_vegetation = [
            f'feature id, garden centroid x, garden centroid y, surface area m², fraction classed as vegetation by {metric_name}'
            'osgb2456, 12.34, 34.56, 23.455, 0.7',
            'osgb2426, 18.34, 49.56, 7.27, 0.23',
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_1of1-imageSource-{metric_name}-vegetation.csv', expected_vegetation)

        expected_toid2uprn = [
            'feature id, feature uprn',
            'osgb2456, 123',
            'osgb2456, 124',
            'osgb2426, 74',
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_1of1-imageSource-{metric_name}-toid2uprn.csv', expected_toid2uprn)

        bulk_recombine.main(['--output-folder', self.output_folder,
                             '--results-folder', self.results_folder,
                             '--index', metric_name,
                             '--loader', 'imageSource'])

        output_summary_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-summary.txt'
        self.assertTrue(os.path.exists(output_summary_file_name))
        with open(output_summary_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_summary)

        output_vegetation_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-vegetation.csv'
        self.assertTrue(os.path.exists(output_vegetation_file_name))
        with open(output_vegetation_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_vegetation)

        output_toid2uprn_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-toid2uprn.csv'
        self.assertTrue(os.path.exists(output_toid2uprn_file_name))
        with open(output_toid2uprn_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_toid2uprn)

    def test_recombine_handles_two_summary_files(self):
        metric_name = 'metric'
        surface_area_m2_1 = 12345.12
        vegetation_surface_area_m2_1 = 4321.93
        vegetation_percentage_1 = vegetation_surface_area_m2_1 / surface_area_m2_1 * 100.0
        self.create_text_file(
            self.results_folder + f'jsonname_1of2-imageSource-{metric_name}-summary.txt',
            [f'Total surface area: {surface_area_m2_1:,.2f}m²',
             f'Total vegetation surface area from {metric_name}: {vegetation_surface_area_m2_1:,.2f}m²'
             f' ({vegetation_percentage_1:.1f}%)',
             'Garden centroid...'])
        vegetation_1 = [
            f'feature id, garden centroid x, garden centroid y, surface area m², fraction classed as vegetation by {metric_name}',
            'osgb2456, 12.34, 34.56, 23.455, 0.7',
            'osgb2426, 18.34, 49.56, 7.27, 0.23',
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_1of2-imageSource-{metric_name}-vegetation.csv', vegetation_1)
        toid2uprn_1 = [
            'feature id, feature uprn',
            'osgb2456, 123',
            'osgb2456, 124',
            'osgb2426, 74',
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_1of2-imageSource-{metric_name}-toid2uprn.csv', toid2uprn_1)

        surface_area_m2_2 = 345.12
        vegetation_surface_area_m2_2 = 321.93
        vegetation_percentage_2 = vegetation_surface_area_m2_2 / surface_area_m2_2 * 100.0
        self.create_text_file(
            self.results_folder + f'jsonname_2of2-imageSource-{metric_name}-summary.txt',
            [f'Total surface area: {surface_area_m2_2:,.2f}m²',
             f'Total vegetation surface area from {metric_name}: {vegetation_surface_area_m2_2:,.2f}m²'
             f' ({vegetation_percentage_2:.1f}%)',
             'Garden centroid...'])
        vegetation_2 = [
            f'feature id, garden centroid x, garden centroid y, surface area m², fraction classed as vegetation by {metric_name}',
            'osgb956, 8.94, 65.76, 19.2, 0.342',
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_2of2-imageSource-{metric_name}-vegetation.csv', vegetation_2)
        toid2uprn_2 = [
            'feature id, feature uprn',
            'osgb956, 72',
            'osgb956, 6'
            ]
        self.create_text_file(
            self.results_folder + f'jsonname_2of2-imageSource-{metric_name}-toid2uprn.csv', toid2uprn_2)

        total_surface_area_m2 = surface_area_m2_1 + surface_area_m2_2
        total_vegetation_surface_area_m2 = vegetation_surface_area_m2_1 + vegetation_surface_area_m2_2
        total_vegetation_percentage = total_vegetation_surface_area_m2 / total_surface_area_m2 * 100.0
        expected_summary = [
            f'Total surface area: {total_surface_area_m2:,.2f}m²',
            f'Total vegetation surface area from {metric_name}: {total_vegetation_surface_area_m2:,.2f}m²'
            f' ({total_vegetation_percentage:.1f}%)'
        ]
        expected_vegetation = vegetation_1 + vegetation_2[1:]
        expected_toid2uprn = toid2uprn_1 + toid2uprn_2[1:]

        bulk_recombine.main(['--output-folder', self.output_folder,
                             '--results-folder', self.results_folder,
                             '--index', metric_name,
                             '--loader', 'imageSource'])

        output_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-summary.txt'
        self.assertTrue(os.path.exists(output_file_name))
        with open(output_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_summary)

        output_vegetation_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-vegetation.csv'
        self.assertTrue(os.path.exists(output_vegetation_file_name))
        with open(output_vegetation_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_vegetation)

        output_toid2uprn_file_name = self.output_folder + f'jsonname_imageSource-{metric_name}-toid2uprn.csv'
        self.assertTrue(os.path.exists(output_toid2uprn_file_name))
        with open(output_toid2uprn_file_name, 'r') as f:
            lines = f.read().splitlines()
            self.assertListEqual(lines, expected_toid2uprn)

