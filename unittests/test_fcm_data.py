import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
from fcmbci.data_processor.fcm_data import FcmDataProcessor
from fcmbci.data_processor.process_functions import *
import itertools
import collections
import pandas as pd 
import xlrd
import warnings


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.fcm = FcmDataProcessor(linguistic_terms=['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])
        self.column_names = ['VL', 'L', 'M', 'H', 'VH']
        self.keys = self.column_names

    def test_instantiate(self):
        self.assertEqual(self.fcm.linguistic_terms, [i.lower() for i in ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH']],
                                msg='Error in the creating an instance of the class!')

    def test_read_excel(self):    
        self.fcm.read_xlsx(filepath=os.path.abspath('unittests/test_cases/data_test.xlsx'), column_names=self.column_names)

        self.assertIsNotNone(self.fcm.data)
        self.assertIsInstance(self.fcm.data, collections.OrderedDict)
        self.assertEqual(len(self.fcm.data), 6)
        self.assertEqual(len(self.fcm.data['Expert_1'].columns), 7)

    def test_read_json(self):
        self.fcm.read_json(filepath=os.path.abspath('unittests/test_cases/data_test.json'),
                                keys=self.keys)
                                
        self.assertIsNotNone(self.fcm.data)
        self.assertIsInstance(self.fcm.data, collections.OrderedDict)
        self.assertEqual(len(self.fcm.data), 2)
        self.assertEqual(len(self.fcm.data['Expert_1'].columns), 8) # plus the NA column.

    # def test_atumf(self):
    #     lt = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH']
    #     results = self.fcm.automf(self.fcm.universe, 
    #                                 linguistic_terms=lt)
    #     # Checks whether the returned object is of type dict.
    #     self.assertEqual(type(results), dict)

    #     # Checks whether the MFs are generated for all the linguistic terms.
    #     self.assertEqual(len(results.keys()), len(lt))

    #     # Check whether the values of the MF are within the fuzzy range 0,1.
    #     res_max = max(list(itertools.chain(*[list(results[i]) for i in results])))
    #     res_min = min(list(itertools.chain(*[list(results[i]) for i in results])))
        
    #     self.assertLessEqual(res_max, 1)
    #     self.assertGreaterEqual(res_min, 0)
    
    # def test_activate(self):
    #     res = self.fcm.automf(self.fcm.universe, 
    #                             linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])
    
    #     act = self.fcm.activate({'M': 0.16, 'H': 0.5, 'VH': 0.33}, res)

    #     # Checks whether the activated membership functions are within the fuzzy range 0,1.
    #     for i in act:
    #         self.assertLessEqual(max(act[i]), 1)
    #         self.assertGreaterEqual(min(act[i]), 0)

    # def test_aggregate(self):
    #     res = self.fcm.automf(self.fcm.universe,
    #                             linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])
    #     act = self.fcm.activate({'M': 0.16, 'H': 0.5, 'VH': 0.33}, res)
    #     aggr = self.fcm.aggregate(act)
        
    #     # the length of the aggregated function should be equal to that of the universe.
    #     self.assertEqual(len(aggr), len(self.fcm.universe))
        
    #     # Checks whether the aggregated function is in the fuzzy range. 
    #     self.assertLessEqual(max(aggr), 1)
    #     self.assertGreaterEqual(min(aggr), 0)

    # def test_defuzzify(self):
    #     res = self.fcm.automf(self.fcm.universe,
    #                             linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'])
    #     act = self.fcm.activate({'M': 0.16, 'H': 0.5, 'VH': 0.33}, res)
    #     aggr = self.fcm.aggregate(act)
        
    #     defuz_res = self.fcm.defuzzify(self.fcm.universe, aggr, method = 'centroid')
    #     self.assertAlmostEqual(defuz_res, 0.70, 2)
    
    # def test_gen_weights(self):
        # self.fcm.gen_weights()
        # weights = self.fcm.causal_weights
        
        # self.assertLessEqual(max(weights.max()), 1)
        # self.assertGreaterEqual(min(weights.min()), -1)


    # def test_weight_edge_list(self):
    #     self.fcm.gen_weights_list(self.data_lst)
    #     weights = self.fcm.causal_weights
        
    #     self.assertLessEqual(max(weights.max()), 1)
    #     self.assertGreaterEqual(min(weights.min()), -1)

    # def test_valence_check(self):
    #     res1 = valence_check('-VH')
    #     res2 = valence_check('VH')
    #     res3 = valence_check(0)

    #     self.assertEqual(res1, -1)
    #     self.assertEqual(res2, 1)
    #     self.assertEqual(res3, 0)   
    

if __name__ == '__main__':
    unittest.main()