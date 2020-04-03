import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
from fcmbci.data_processor.fcm_data import FcmDataProcessor
import itertools
import pandas as pd 



class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.fcm = FcmDataProcessor()

    def test_read_excel(self):
        self.fcm.read_xlsx('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/jupyter_prototype/sample_test.xlsx')
        data = self.fcm.data
        
        # Checks whether the data was successfully read or not. 
        self.assertIsNotNone(data)
    
    def test_atumf(self):
        results = self.fcm.automf(self.fcm.universe, linguistic_terms=['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])

        # Checks whether the returned object is of type dict.
        self.assertEqual(type(results), dict)

        # Checks whether the MFs are generated for all the linguistic terms.
        self.assertEqual(len(results.keys()), 10)

        # Check whether the values of the MF are within the fuzzy range 0,1.
        res_max = max(list(itertools.chain(*[list(results[i]) for i in results])))
        res_min = min(list(itertools.chain(*[list(results[i]) for i in results])))
        
        self.assertLessEqual(res_max, 1)
        self.assertGreaterEqual(res_min, 0)
    
    # def test_activate(self):
    #     mf = self.fcm.automf(linguistic_terms = ['VL', 'L'])
    #     activated = self.fcm.activate({'VL': 0.66, 'L': 0.33}, mf)
        
        # Checks whether the activated membership functions are within the fuzzy range 0,1.


if __name__ == '__main__':
    unittest.main()