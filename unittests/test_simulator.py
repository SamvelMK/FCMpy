import unittest
import pandas as pd
from fcmpy.simulator.simulator import FcmSimulator

class TestSimulator(unittest.TestCase):
    
    def setUp(self):

        C1 = [0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.8]
        C2 = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5]
        C3 = [0.0, 0.7, 0.0, 0.0, 0.9, 0.0, 0.4, 0.1]
        C4 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
        C5 = [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.3]
        C6 = [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        C7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.9]
        C8 =[0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.0]
        df = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                            columns=['C1','C2','C3','C4','C5','C6','C7','C8'])
        self.weight_matrix = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                        columns=['C1','C2','C3','C4','C5','C6','C7','C8'])

        self.init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                        'C6': 0, 'C7': 0, 'C8': 0}
        self.sim = FcmSimulator()
        
    def test_simulation(self):
        res_k = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='sigmoid', inference='kosko', thresh=0.001, iterations=50, l=1)
        res_mK = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)
        res_r = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='sigmoid', inference='rescaled', thresh=0.001, iterations=50, l=1)

        eql_k = res_k.loc[len(res_k)-1] # equilibruim for Kosko's menthod
        eql_mK = res_mK.loc[len(res_mK)-1] # equilibruim for Kosko's menthod
        eql_r = res_r.loc[len(res_r)-1] # equilibruim for rescaled menthod

        # test the results against the one presented in the fcm inference package in R by Dikopoulou & Papageorgiou.
        equilibrium_mK = [0.7258851, 0.7907061, 0.7694508, 0.8124733, 0.8192938, 0.8399006, 0.9099403, 0.9557739]
        equilibrium_k = [0.5481291, 0.6004012, 0.5814756, 0.620872, 0.6279569, 0.6653852, 0.7617789, 0.8416584]
        equilibrium_r = [0.4998427, 0.4985174, 0.4989474, 0.4984212, 0.4980938, 0.4885541, 0.4855651, 0.4853266]

        # check if the initial equilibrium = to the one by Dikopoulou & Papageorgiou in R.
        self.assertEqual([round(i, 4) for i in eql_k], [round(i, 4) for i in equilibrium_k])
        self.assertEqual([round(i, 4) for i in eql_mK], [round(i, 4) for i in equilibrium_mK])
        self.assertEqual([round(i, 4) for i in eql_r], [round(i, 4) for i in equilibrium_r])

    def test_simulation_bi(self):
        res_mK = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='bivalent', inference='mKosko', thresh=0.001, iterations=50, l=1)
        self.assertEqual(len(set(res_mK.values.flatten())), 2)
        self.assertEqual(max(res_mK.values.flatten()), 1)
        self.assertEqual(min(res_mK.values.flatten()), 0)

    def test_simulation_tri(self):
        init_state = self.init_state.copy()
        init_state['C1'] = -1
        res_mK = self.sim.simulate(initial_state=init_state, weight_matrix=self.weight_matrix, transfer='trivalent', inference='mKosko', thresh=0.001, iterations=50, l=1)
        self.assertEqual(len(set(res_mK.values.flatten())), 3)
        self.assertEqual(max(res_mK.values.flatten()), 1)
        self.assertEqual(min(res_mK.values.flatten()), -1)

    def test_simulation_tanh(self):
        init_state = self.init_state.copy()
        init_state['C1'] = -1
        res_mK = self.sim.simulate(initial_state=init_state, weight_matrix=self.weight_matrix, transfer='tanh', inference='mKosko', thresh=0.001, iterations=50, l=1)
        self.assertEqual(max(res_mK.values.flatten()), 1)
        self.assertEqual(min(res_mK.values.flatten()), -1)

    def test_stableConcepts(self):
        self.weight_matrix['C1'] = 0
        res_k = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='sigmoid', inference='kosko', thresh=0.001, iterations=50, l=1)
        self.assertEqual(len(set(res_k['C1'])), 1)

    def test_outputConcepts(self):
        out = ['C4', 'C1']
        res_mK = self.sim.simulate(initial_state=self.init_state, weight_matrix=self.weight_matrix, transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1, output_concepts=out)
        _ = res_mK[out]
        residual = max(abs(_.loc[len(_)-1] - _.loc[len(_) - 2]))
        self.assertLessEqual(residual, 0.001)
        self.assertEqual([0.812650, 0.726125], list(_.loc[len(_)-1].values.round(6)))

if __name__ == '__main__':
    unittest.main()