import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
from fcmbci.simulator.simulator import FcmSimulator
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
        df = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                            columns=['C1','C2','C3','C4','C5','C6','C7','C8'])
        self.df = df
        self.init = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                        'C6': 0, 'C7': 0, 'C8': 0}

        self.sim = FcmSimulator(self.init, self.df)

    def test_init(self):
        res_k = self.sim.simulate(self.init, self.df, inference = 'k')
        res_r = self.sim.simulate(self.init, self.df, inference = 'r')

        eql_k = res_k.loc[len(res_k)-1] # equilibruim for Kosko's menthod
        eql_r = res_r.loc[len(res_r)-1] # equilibruim for rescaled menthod

        equilibrium_mk = [0.7258851, 0.7907061, 0.7694508, 0.8124733, 0.8192938, 0.8399006, 0.9099403, 0.9557739]
        equilibrium_k = [0.5481291, 0.6004012, 0.5814756, 0.620872, 0.6279569, 0.6653852, 0.7617789, 0.8416584]
        equilibrium_r = [0.4998427, 0.4985174, 0.4989474, 0.4984212, 0.4980938, 0.4885541, 0.4855651, 0.4853266]

        # check if the initial equilibrium = to the one by Dikopoulou & Papageorgiou in R.
        self.assertEqual(list(round(self.sim.initial_equilibrium, 7)), equilibrium_mk)
        self.assertEqual(list(round(eql_k, 7)), equilibrium_k)
        self.assertEqual(list(round(eql_r, 7)), equilibrium_r)

        # check if the number of iterations to reach an equilibrium = to the one by Dikopoulou & Papageorgiou in R. 
        self.assertEqual(len(self.sim.scenarios['initial_state']), 7) # Modified Kosko
        self.assertEqual(len(res_k), 7) # Kosko
        self.assertEqual(len(res_r), 49) # Rescaled

    def test_scenarios(self):
        self.sim.test_scenario('mk', state_vector= {'C1' : 0, 'C2': 1}, weights = self.df)

        equilibrium_mk = [0.7255749, 0.7905062, 0.7691449, 0.8121284, 0.8190922, 0.839738, 0.9098928, 0.9557094]
        eql_mk = self.sim.scenarios['mk'].loc[len(self.sim.scenarios['mk']) -1]

if __name__ == '__main__':
    unittest.main()