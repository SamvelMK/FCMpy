import unittest
from fcm_data import FcmDataProcessor
import itertools

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.fcm = FcmDataProcessor()
        self.fcm.read_xlsx('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/jupyter_prototype/sample_test.xlsx')

    def test_read_excel(self):
        self.fcm.read_xlsx('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/jupyter_prototype/sample.xlsx')
        data = self.fcm.data
        
        # Checks whether data object was created or not. 
        self.assertIsNotNone(data)
    
    def test_read_csv(self):
        self.fcm.read_csv('C:/PhD/FCM_Projects/FCM_Python/FCM_BCI/jupyter_prototype/sample_csv.csv', sep = ';')
        data = self.fcm.data

        # Checks whether data object was created or not. 
        self.assertIsNotNone(data)
    
    def test_atumf(self):
        results = self.fcm.automf(linguistic_terms=['+VH', 'H', 'VL'])

        # Checks whether the MFs are generated for all the linguistic terms.
        self.assertEqual(len(results.keys()), 3)

        # Checks whether the returned object is of type dict.
        self.assertEqual(type(results), dict)

        # Check whether the values of the MF are within the fuzzy range 0,1.
        res_max = max(list(itertools.chain(*[list(results[i]) for i in results])))
        res_min = min(list(itertools.chain(*[list(results[i]) for i in results])))
        self.assertLessEqual(res_max, 1)
        self.assertGreaterEqual(res_min, 0)
    
    def test_activate(self):
        mf = self.fcm.automf(linguistic_terms = ['VL', 'L'])
        activated = self.fcm.activate({'VL': 0.66, 'L': 0.33}, mf)
        
        # Checks whether the activated membership functions are within the fuzzy range 0,1.
        res_max = max(list(itertools.chain(*[list(activated[i]) for i in activated])))
        res_min = min(list(itertools.chain(*[list(activated[i]) for i in activated])))
        self.assertLessEqual(res_max, 1)
        self.assertGreaterEqual(res_min, 0)

    def test_aggregate(self):
        # mf = self.fcm.automf(linguistic_terms = ['VH','H', 'M', 'L', 'VL'])
        # activated = self.fcm.activate({'H': 0.16666666666666666, 'M': 0.5, 'VH': 0.3333333333333333}, mf)

        # self.aggregated = self.fcm.aggregate(activated)
        pass
    
    def test_defuzzify(self):
        mf = self.fcm.automf(linguistic_terms = ['VL','L', 'M', 'H', 'VH'])
        activated = self.fcm.activate({'H': 0.16666666666666666, 'M': 0.5, 'VH': 0.3333333333333333}, mf)
        aggregated = self.fcm.aggregate(activated)
        defuzz_value = self.fcm.defuzzify(self.fcm.universe, aggregated, method = 'centroid')
        print(defuzz_value)
        self.assertAlmostEqual(round(defuzz_value, 2), 0.61)

    def test_valence_check(self):
        val = self.fcm.valence_check('-VL')
        self.assertEqual(val, -1)

    def test_generate_edge_weights(self):
        
        # self.fcm.generate_edge_weights(linguistic_terms = ['VL', 'L', 'M', 'H', 'VH'],
        #                                             method = 'centroid')
        # print(self.fcm.causal_weights)
        # self.assertEqual(weights, true_weights)
        pass


if __name__ == '__main__':
    unittest.main()