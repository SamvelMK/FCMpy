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

        weight_matrix = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                            columns=['C1','C2','C3','C4','C5','C6','C7','C8'])

        init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                            'C6': 0, 'C7': 0, 'C8': 0}

        self.inter = Intervention(initial_state=init_state, weight_matrix=weight_matrix, transfer='sigmoid', inference='mKosko', 
                                        thresh=0.001, iterations=100, l=1)

    def test_addIntervention(self):
        # Check if the interventions were properly added to the constructor.
        self.inter.add_intervention('intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
        self.inter.add_intervention('intervention_2', weights={'C1':-.5}, effectiveness=1)
        self.inter.add_intervention('intervention_3', weights={'C1':-1}, effectiveness=1)

        intervations = ['intervention_1', 'intervention_2', 'intervention_3']
        nInter = len(set(intervations) ^ set(self.inter.interventions.keys()))
        self.assertEqual(nInter, 0, msg="The interventions were not added properly to the constructor!")

    def test_removeIntervention(self):
        # Check if the interventions were properly added to the constructor.
        self.inter.add_intervention('intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
        self.inter.add_intervention('intervention_2', weights={'C1':-.5}, effectiveness=1)
        self.inter.add_intervention('intervention_3', weights={'C1':-1}, effectiveness=1)

        self.inter.remove_intervention('intervention_1')
        intervations = ['intervention_2', 'intervention_3']
        nInter = len(set(intervations) ^ set(self.inter.interventions.keys()))
        self.assertEqual(nInter, 0, msg="The intervention was not removed from the constructor!")

    def test_testIntervention(self):
        # Check if the itest_intervention runs properly.
        # Check if the stable concept (intervetion in this case) is indeed stable.
        self.inter.add_intervention('intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
        self.inter.test_intervention('intervention_1')
        self.assertEqual(len(set(self.inter.test_results['intervention_1']['intervention'])), 1)

if __name__ == '__main__':
    unittest.main()