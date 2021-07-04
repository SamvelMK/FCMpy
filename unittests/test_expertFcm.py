import sys, os
import unittest
from fcmpy.expert_fcm.expert_based_fcm import ExpertFcm
import itertools
import collections
import pandas as pd 
import xlrd
import warnings
import functools
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.fcm = ExpertFcm()
        self.nExpert = 6
        self.activation_param = {'M': 0.16, 'H': 0.5, 'VH': 0.33}
        self.fcm.universe = np.arange(-1, 1.001, 0.001)

        self.fcm.linguistic_terms = {
                                    '-VH': [-1, -1, -0.75],
                                    '-H': [-1, -0.75, -0.50],
                                    '-M': [-0.75, -0.5, -0.25], 
                                    '-L': [-0.5, -0.25, 0],
                                    '-VL': [-0.25, 0, 0],
                                    'NA': [-0.001, 0, 0.001],
                                    '+VL': [0, 0, 0.25],
                                    '+L': [0, 0.25, 0.50],
                                    '+M': [0.25, 0.5, 0.75],
                                    '+H': [0.5, 0.75, 1],
                                    '+VH':  [0.75, 1, 1]
                                    }
        
        self.fcm.fuzzy_membership = self.fcm.automf(method='trimf')

    def test_readExcel(self):
        data = self.fcm.read_data(file_path=os.path.abspath('unittests/test_cases/data_test.xlsx'), check_consistency=False, engine='openpyxl') 
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), self.nExpert)

    def test_readExcelUnsure(self):
        data = self.fcm.read_data(file_path=os.path.abspath('unittests/test_cases/data_test_unsure.xlsx'), check_consistency=False, engine='openpyxl')
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), self.nExpert)

    def test_readJson(self):
        data = self.fcm.read_data(file_path=os.path.abspath('unittests/test_cases/data_test.json'), check_consistency=False)
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), 2)

    def test_readJsonUnsure(self):
        data = self.fcm.read_data(file_path=os.path.abspath('unittests/test_cases/data_test_unsure.json'), check_consistency=False)
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), 2)

    def test_readCsv(self):
        self.fcm.linguistic_terms = {
                                    '-VH':  [-1, -1, -0.75],
                                    '-H':  [-1, -0.75, -0.50],
                                    '-M': [-0.75, -0.5, -0.25], 
                                    '-L': [-0.5, -0.25, 0],
                                    '-VL': [-0.25, 0, 0],
                                    'No Causality': [-0.001, 0, 0.001],
                                    '+VL': [0, 0, 0.25],
                                    '+L': [0, 0.25, 0.50],
                                    '+M': [0.25, 0.5, 0.75],
                                    '+H': [0.5, 0.75, 1],
                                    '+VH':  [0.75, 1, 1]
                                    }
        data = self.fcm.read_data(os.path.abspath('unittests/test_cases/data_test.csv'), sep_concept='->', csv_sep=';')
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), self.nExpert)
        
    def test_readCsvUnsure(self):
        self.fcm.linguistic_terms = {
                                    '-VH':  [-1, -1, -0.75],
                                    '-H':  [-1, -0.75, -0.50],
                                    '-M': [-0.75, -0.5, -0.25], 
                                    '-L': [-0.5, -0.25, 0],
                                    '-VL': [-0.25, 0, 0],
                                    'No Causality': [-0.001, 0, 0.001],
                                    '+VL': [0, 0, 0.25],
                                    '+L': [0, 0.25, 0.50],
                                    '+M': [0.25, 0.5, 0.75],
                                    '+H': [0.5, 0.75, 1],
                                    '+VH':  [0.75, 1, 1]
                                    }
        data = self.fcm.read_data(os.path.abspath('unittests/test_cases/data_test_unsure.csv'), sep_concept='->', csv_sep=';')
        self.assertIsInstance(data, collections.OrderedDict)
        self.assertEqual(len(data), self.nExpert)

    def test_entropy(self):
        data = self.fcm.read_data(os.path.abspath('unittests/test_cases/data_test_entropy.xlsx'))
        entropy = self.fcm.entropy(data=data, method = 'entropy')
        self.assertGreaterEqual(min(entropy['Entropy']), 0)
        self.assertEqual(max(entropy['Entropy']), 0)
        
    def test_entropyUnsure(self):
        data = self.fcm.read_data(os.path.abspath('unittests/test_cases/data_test_entropy_unsure.xlsx'))
        entropy = self.fcm.entropy(data=data, method = 'entropy')
        self.assertEqual(round(max(entropy['Entropy']), 2), 0.65)

    def test_automf(self):
        mf  = self.fcm.automf(method = 'trimf')
        self.assertEqual(type(mf), dict)
        self.assertEqual(len(mf.keys()), len(self.fcm.linguistic_terms))

        # Check whether the values of the MF are within the fuzzy range 0,1.
        res_max = max(list(itertools.chain(*[list(mf[i]) for i in mf])))
        res_min = min(list(itertools.chain(*[list(mf[i]) for i in mf])))
        self.assertLessEqual(res_max, 1)
        self.assertGreaterEqual(res_min, 0)

        # check whether the generated universe is in the range -1,1 (default)
        self.assertLessEqual(round(max(self.fcm.universe), 5), 1)
        self.assertGreaterEqual(round(min(self.fcm.universe), 5), -1)

    def test_activate(self):
        mf  = self.fcm.fuzzy_membership
        act_M = self.fcm.fuzzy_implication(mf['+m'], weight=0.16, method ='Larsen')
        act_H = self.fcm.fuzzy_implication(mf['+h'], weight=0.5, method ='Larsen')
        act_VH = self.fcm.fuzzy_implication(mf['+vh'], weight= 0.33, method ='Larsen')
        activated = {'+m' : act_M, '+h' : act_H, '+vh' : act_VH}

        # Checks whether the activated membership functions are within the fuzzy range 0,1.
        for i in activated:
            self.assertLessEqual(max(activated[i]), 1)
            self.assertGreaterEqual(min(activated[i]), 0)

    def test_aggregate(self):
        mf  = self.fcm.fuzzy_membership
        act_M = self.fcm.fuzzy_implication(mf['+m'], weight=0.16, method ='Larsen')
        act_H = self.fcm.fuzzy_implication(mf['+h'], weight=0.5, method ='Larsen')
        act_VH = self.fcm.fuzzy_implication(mf['+vh'], weight= 0.33, method ='Larsen')
        activated = {'+m' : act_M, '+h' : act_H, '+vh' : act_VH}

        aggr = functools.reduce(lambda x,y: np.fmax(x,y), [activated[i] for i in activated.keys()])
        
        # the length of the aggregated function should be equal to that of the universe.
        self.assertEqual(len(aggr), len(self.fcm.universe))
        
        # Checks whether the aggregated function is in the fuzzy range. 
        self.assertLessEqual(max(aggr), 1)
        self.assertGreaterEqual(min(aggr), 0)

    def test_defuzzify(self):
        mf  = self.fcm.fuzzy_membership
        act_M = self.fcm.fuzzy_implication(mf['+m'], weight=0.16, method ='Larsen')
        act_H = self.fcm.fuzzy_implication(mf['+h'], weight=0.5, method ='Larsen')
        act_VH = self.fcm.fuzzy_implication(mf['+vh'], weight= 0.33, method ='Larsen')
        activated = {'+m' : act_M, '+h' : act_H, '+vh' : act_VH}

        aggr = functools.reduce(lambda x,y: np.fmax(x,y), [activated[i] for i in activated.keys()])

        dfuz = self.fcm.defuzz(x=self.fcm.universe, mfx=aggr)    
        self.assertAlmostEqual(dfuz, 0.72, 2)
    
    def test_build(self):
        self.fcm.fuzzy_membership = self.fcm.automf(method='trimf')
        
        data = self.fcm.read_data(file_path=os.path.abspath('unittests/test_cases/data_test.xlsx'), check_consistency=False, engine='openpyxl')
        weight_matrix = self.fcm.build(data=data, implication_method='Larsen')
        self.assertLessEqual(max(weight_matrix.max()), 1)
        self.assertGreaterEqual(min(weight_matrix.min()), -1)    

if __name__ == '__main__':
    unittest.main()