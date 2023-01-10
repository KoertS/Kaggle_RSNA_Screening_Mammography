import unittest

import pandas as pd

from src.data.datagenerator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        # create a sample dataframe with a minority class (cancer = 1) and majority class (cancer = 0)
        self.dataframe = pd.DataFrame({'patient_id': ['1', '2', '3', '4', '5', '6'],
                                       'image_id': ['a', 'b', 'c', 'd', 'e', 'f'],
                                       'laterality': ['l'] * 6,
                                       'view': ['MLO'] * 6,
                                       'cancer': [1, 0, 0, 0, 1, 0]})

    def test_oversampling(self):
        count_cancer = sum(self.dataframe['cancer'])

        factors = [0, 1, 2, 10]
        for factor in factors:
            data_gen = DataGenerator(dataframe=self.dataframe,
                                     path_images='path/to/images/',
                                     input_size=1,
                                     oversampling_factor=factor)
            # check that the number of minority class samples
            if factor == 0:
                self.assertEqual(sum(data_gen.dataframe['cancer']), count_cancer)
            else:
                self.assertEqual(sum(data_gen.dataframe['cancer']), count_cancer * factor)


if __name__ == '__main__':
    unittest.main()
