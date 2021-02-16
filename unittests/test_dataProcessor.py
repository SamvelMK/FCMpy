import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
from fcmbci.data_processor.data_processor import DataProcessor
import itertools
import collections
import pandas as pd 
import xlrd
import warnings


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.fcm = DataProcessor(linguistic_terms=['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])
        self.nExpert = 6
        self.activation_param = {'M': 0.16, 'H': 0.5, 'VH': 0.33}
        
    def test_instantiate(self):
        # Universe of discourse        
        self.assertTrue(round(self.fcm.universe.max(), 2) == 1, msg="The universe of discourse is out of bounds.")
        self.assertTrue(round(self.fcm.universe.min(), 2) == -1, msg="The universe of discourse is out of bounds.")
    
    def test_readExcel(self):
         self.fcm.read_xlsx(filepath=os.path.abspath('unittests/test_cases/data_test.xlsx'), check_consistency=False)
         self.assertIsInstance(self.fcm.data, collections.OrderedDict)
         self.assertEqual(len(self.fcm.data), self.nExpert)

    def test_readExcelUnsure(self):
        self.fcm.read_xlsx(filepath=os.path.abspath('unittests/test_cases/data_test_unsure.xlsx'), check_consistency=False)
        self.assertIsInstance(self.fcm.data, collections.OrderedDict)
        self.assertEqual(len(self.fcm.data), self.nExpert)

    def test_readJson(self):
        self.fcm.read_json(filepath=os.path.abspath('unittests/test_cases/data_test.json'), check_consistency=False)
        self.assertIsInstance(self.fcm.data, collections.OrderedDict)
        self.assertEqual(len(self.fcm.data), 2)

    def test_readJsonUnsure(self):
        self.fcm.read_json(filepath=os.path.abspath('unittests/test_cases/data_test_unsure.json'), check_consistency=False)
        self.assertIsInstance(self.fcm.data, collections.OrderedDict)
        self.assertEqual(len(self.fcm.data), 2)

    def test_readCsv(self):
        fcm = DataProcessor(linguistic_terms=['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'], no_causality='no causality')
        fcm.read_csv('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/PyFcmBci/unittests/test_cases/data_test.csv', sepConcept='->', csv_sep=';')
        self.assertIsInstance(fcm.data, collections.OrderedDict)
        self.assertEqual(len(fcm.data), self.nExpert)

    def test_atumf(self):
        results = self.fcm.automf()
        # Checks whether the returned object is of type dict.
        self.assertEqual(type(results), dict)

        # Checks whether the MFs are generated for all the linguistic terms.
        self.assertEqual(len(results.keys()), len(self.fcm.linguistic_terms))

        # Check whether the values of the MF are within the fuzzy range 0,1.
        res_max = max(list(itertools.chain(*[list(results[i]) for i in results])))
        res_min = min(list(itertools.chain(*[list(results[i]) for i in results])))
        
        self.assertLessEqual(res_max, 1)
        self.assertGreaterEqual(res_min, 0)
    
    def test_activate(self):
        res = self.fcm.automf()
        act = self.fcm.activate(mf= res, activation_input=self.activation_param)

        # Checks whether the activated membership functions are within the fuzzy range 0,1.
        for i in act:
            self.assertLessEqual(max(act[i]), 1)
            self.assertGreaterEqual(min(act[i]), 0)

    def test_aggregate(self):
        res = self.fcm.automf()
        act = self.fcm.activate(mf=res, activation_input=self.activation_param)
        aggr = self.fcm.aggregate(act)
        
        # the length of the aggregated function should be equal to that of the universe.
        self.assertEqual(len(aggr), len(self.fcm.universe))
        
        # Checks whether the aggregated function is in the fuzzy range. 
        self.assertLessEqual(max(aggr), 1)
        self.assertGreaterEqual(min(aggr), 0)

    def test_defuzzify(self):
        res = self.fcm.automf()
        act = self.fcm.activate(mf=res, activation_input=self.activation_param)
        aggr = self.fcm.aggregate(act)
        
        defuz_res = self.fcm.defuzzify(aggr, method = 'centroid')
        self.assertAlmostEqual(defuz_res, 0.72, 2)
    
    def test_gen_weights(self):
        fcm = DataProcessor(linguistic_terms=['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'], no_causality='NA')
        fcm.read_xlsx(filepath=os.path.abspath('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/PyFcmBci/unittests/test_cases/data_test.xlsx'), check_consistency=False)
        fcm.gen_weights()
        weight_matrix = fcm.weight_matrix
        
        self.assertLessEqual(max(weight_matrix.max()), 1)
        self.assertGreaterEqual(min(weight_matrix.min()), -1)    

if __name__ == '__main__':
    unittest.main()