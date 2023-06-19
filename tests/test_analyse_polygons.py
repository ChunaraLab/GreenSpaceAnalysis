import os

from pyfakefs.fake_filesystem_unittest import TestCase

from green_spaces import analyse_polygons


class TestAnalysePolygons(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_empty_warnings_do_not_produce_warnings_file(self):
        root_folder = '/fish-tmp/'
        os.makedirs(root_folder)

        map_loader_name = 'dummyMapLoader'
        analyse_polygons.report_feature_analysis([], [], map_loader_name, root_folder, 'CRS', [])

        self.assertFalse(os.path.exists(root_folder + map_loader_name + '-warnings.txt'))

    def test_some_warnings_does_produce_warnings_file(self):
        root_folder = '/fish-tmp/'
        os.makedirs(root_folder)
        map_loader_name = 'dummyMapLoader'
        warnings = ['Warning #1', 'Warning #2']

        analyse_polygons.report_feature_analysis([], [], map_loader_name, root_folder, 'CRS', warnings)

        self.assertTrue(os.path.exists(root_folder + map_loader_name + '-warnings.txt'))
        with open(root_folder + map_loader_name + '-warnings.txt') as f:
            actual_warnings = f.read().splitlines()
            self.assertListEqual(warnings, actual_warnings)
