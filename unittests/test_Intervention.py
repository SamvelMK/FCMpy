import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
from fcmbci.intervention.intervention import Intervention
import pandas as pd 

class TestFcmSimulator(unittest.TestCase):
    
    def setUp(self):

        C1 = [0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.8]
        C2 = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5]
        C3 = [0.0, 0.7, 0.0, 0.0, 0.9, 0.0, 0.4, 0.1]
        C4 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
        C5 = [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.3]
        C6 = [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        C7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.9]
        C8 =[0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.0]

        weight_mat = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                            columns=['C1','C2','C3','C4','C5','C6','C7','C8'])

        init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                            'C6': 0, 'C7': 0, 'C8': 0}

        self.inter = Intervention(initial_state=init_state, weights=weight_mat, transfer='sigmoid', inference='mKosko', 
                                        thresh=0.001, iterations=100, l=1)

    def test_addIntervention(self):
        pass

    def test_removeIntervention(self):
        pass

    def test_testIntervention(self):
        pass

if __name__ == '__main__':
    unittest.main()