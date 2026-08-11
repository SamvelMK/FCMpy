import unittest
from fcmpy.simulator.simulator import FcmSimulator
from fcmpy.intervention.intervention import FcmIntervention
import pandas as pd

class TestIntervention(unittest.TestCase):
    
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
        
        self.inter = FcmIntervention(FcmSimulator)
        self.inter.initialize(initial_state=init_state, weight_matrix=weight_matrix, transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)

    def test_addIntervention(self):
        # Check if the interventions were properly added to the constructor.
        self.inter.add_intervention('intervention_1', impact={'C1':-.3, 'C2' : .5}, effectiveness=1)
        self.inter.add_intervention('intervention_2', impact={'C1':-.5}, effectiveness=1)
        self.inter.add_intervention('intervention_3', impact={'C1':-1}, effectiveness=1)

        intervations = ['intervention_1', 'intervention_2', 'intervention_3']
        nInter = len(set(intervations) ^ set(self.inter.interventions.keys()))
        self.assertEqual(nInter, 0, msg="The interventions were not added properly to the constructor!")

    def test_removeIntervention(self):
        # Check if the interventions were properly added to the constructor.
        self.inter.add_intervention('intervention_1', impact={'C1':-.3, 'C2' : .5}, effectiveness=1)
        self.inter.add_intervention('intervention_2', impact={'C4':-.5}, effectiveness=1)
        self.inter.add_intervention('intervention_3', impact={'C5':-1}, effectiveness=1)

        self.inter.remove_intervention('intervention_1')
        intervations = ['intervention_2', 'intervention_3']
        nInter = len(set(intervations) ^ set(self.inter.interventions.keys()))
        self.assertEqual(nInter, 0, msg="The intervention was not removed from the constructor!")

    def test_testIntervention(self):
        # Check if the test_intervention runs properly.
        # Check if the stable concept (intervetion in this case) is indeed stable.
        self.inter.add_intervention('intervention_1', impact={'C1':-.3, 'C2' : .5})
        self.inter.add_intervention('intervention_2', impact={'C4':-.5})
        self.inter.add_intervention('intervention_3', impact={'C5':-1})

        self.inter.test_intervention('intervention_1')
        self.inter.test_intervention('intervention_2')
        self.inter.test_intervention('intervention_3')
        
        self.assertEqual(len(set(self.inter.test_results['intervention_1']['intervention'])), 1)
    
    def test_singleShot(self):
        self.inter.add_intervention('intervention_1', type='single_shot', initial_state = {'C1': 0.9, 'C2' : 0.4})
        self.assertEqual(self.inter.interventions['intervention_1']['state_vector']['C1'], 0.9)
        self.assertEqual(self.inter.interventions['intervention_1']['state_vector']['C2'], 0.4)

    def test_continuousReorderedIndex(self):
        # Regression test: weight_matrix whose row (index) order does not match
        # its column order should not have its edges misattributed to the wrong
        # concept when building a continuous intervention.
        cols = ['A', 'B', 'C']
        data = pd.DataFrame(
            [[0.0, 0.9, 0.0],   # A -> B = 0.9, the only nonzero edge
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            columns=cols, index=cols
        )
        wm_reordered = data.loc[['C', 'A', 'B']]  # physically reordered rows, correct labels

        init_state = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        inter = FcmIntervention(FcmSimulator)
        inter.initialize(initial_state=init_state, weight_matrix=wm_reordered, transfer='sigmoid',
                          inference='mKosko', thresh=0.001, iterations=10, l=1)
        inter.add_intervention('intervention_1', impact={'C': 0.5}, effectiveness=1)

        built_wm = inter.interventions['intervention_1']['weight_matrix']
        self.assertEqual(built_wm.loc['A', 'B'], 0.9, msg="A->B edge was misattributed to the wrong row!")
        self.assertEqual(built_wm.loc['A', 'A'], 0.0)
        self.assertEqual(built_wm.loc['A', 'C'], 0.0)
        self.assertEqual(built_wm.loc['B'][['A', 'B', 'C']].sum(), 0.0, msg="B should have no outgoing edges!")

if __name__ == '__main__':
    unittest.main()