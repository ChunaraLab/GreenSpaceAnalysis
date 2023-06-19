import json
import os

from pyfakefs.fake_filesystem_unittest import TestCase

from scripts import split_geojson


class TestGeoJsonSplitting(TestCase):
    def setUp(self):
        self.setUpPyfakefs()
        self.feature_1 = {"type": "Feature",
                          "properties": {"id": "osgb12345", "uprn": "{54321}", "msoa11cd": "W01",
                                         "msoa11nm": "Fish 033", "lsoa11cd": "W123", "lsoa11nm": "Fish 033A"},
                          "geometry": {"type": "MultiPolygon",
                                       "coordinates": [[[[-3.8, 52.3], [-3.9, 52.4], [-3.9, 52.3], [-3.8, 52.3]]]]}
                          }

        self.feature_2 = {"type": "Feature",
                          "properties": {"id": "osgb12346", "uprn": "{54322}", "msoa11cd": "W02",
                                         "msoa11nm": "Fish 0332", "lsoa11cd": "W1232", "lsoa11nm": "Fish 033A2"},
                          "geometry": {"type": "MultiPolygon",
                                       "coordinates": [[[[-3.8, 50.3], [-3.9, 50.4], [-3.9, 50.3], [-3.8, 50.3]]]]}
                          }

        self.feature_3 = {"type": "Feature",
                          "properties": {"id": "osgb12347", "uprn": "{54323}", "msoa11cd": "W03",
                                         "msoa11nm": "Fish 0331", "lsoa11cd": "W1231", "lsoa11nm": "Fish 033A1"},
                          "geometry": {"type": "MultiPolygon",
                                       "coordinates": [[[[-13.8, 52.3], [-13.9, 52.4], [-13.9, 52.3], [-13.8, 52.3]]]]}
                          }

        self.geojson = {
            "type": "FeatureCollection",
            "name": "22052018_cardiff_residential_gardens",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [
                self.feature_1,
                self.feature_2,
                self.feature_3
            ]
        }

    def test_geojson_not_split_if_smaller_than_or_equal_to_features_per_file(self):
        geojson_file_name = 'small.geojson'
        expected_output_file_name = 'small_1of1.geojson'

        with open(geojson_file_name, 'w') as f:
            json.dump(self.geojson, f)

        split_geojson.main(['-fpf', '3', geojson_file_name])

        self.assertTrue(os.path.exists(expected_output_file_name))

        with open(expected_output_file_name, 'r') as f:
            actual_geojson = json.load(f)

        self.assertDictEqual(self.geojson, actual_geojson)

    def test_geojson_split_if_larger_than_feature_per_file(self):
        geojson_file_name = 'small.geojson'

        expected_output_1_file_name = 'small_1of2.geojson'
        expected_1_geojson = self.geojson.copy()
        expected_1_geojson['features'] = [self.feature_1, self.feature_2]

        expected_output_2_file_name = 'small_2of2.geojson'
        expected_2_geojson = self.geojson.copy()
        expected_2_geojson['features'] = [self.feature_3]

        with open(geojson_file_name, 'w') as f:
            json.dump(self.geojson, f)

        split_geojson.main(['-fpf', '2', geojson_file_name])

        self.assertTrue(os.path.exists(expected_output_1_file_name))
        with open(expected_output_1_file_name, 'r') as f:
            actual_1_geojson = json.load(f)
        self.assertDictEqual(expected_1_geojson, actual_1_geojson)

        self.assertTrue(os.path.exists(expected_output_2_file_name))
        with open(expected_output_2_file_name, 'r') as f:
            actual_2_geojson = json.load(f)
        self.assertDictEqual(expected_2_geojson, actual_2_geojson)
