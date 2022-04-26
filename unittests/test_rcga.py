import unittest
import pandas as pd 
import numpy as np
from fcmpy import FcmSimulator
from fcmpy.store.methodsStore import AuxiliaryStore
from fcmpy.store.methodsStore import NormalizationStore
from fcmpy.store.methodsStore import MatrixErrorStore
from fcmpy.store.methodsStore import InitializationStore
from fcmpy.ml.genetic.evaluation import PopulationEvaluation

class TestRcga(unittest.TestCase):

    def setUp(self):
        w_init = np.asarray([[0,-0.4,-0.25,0,0.3],[0.36,0,0,0,0],
                        [0.45,0,0,0,0],[-0.9,0,0,0,0],
                        [0,0.6,0,0.3,0]])

        self.w_init_df = pd.DataFrame(w_init, columns=['C1', 'C2', 'C3', 'C4', 'C5'],
                                index = ['C1', 'C2', 'C3', 'C4', 'C5'])

        init_state = np.asarray([0.4,0.707,0.607,0.72,0.3])
        self.init_state_d = {'C1' : 0.40, 'C2': 0.7077, 'C3': 0.612, 'C4': 0.717, 'C5': 0.30}
        sim = FcmSimulator()
        self.data = sim.simulate(initial_state=self.init_state_d, weight_matrix=self.w_init_df,
                            transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)

        self.auxiliary = AuxiliaryStore.get('h')
        self.normalization = NormalizationStore.get('L2')
        self.error = MatrixErrorStore.get('stach_error')

    def test_fitness(self):
        matError= self.error.calculate(data_simulated=self.data, data=(self.data), p=2)
        normalized = self.normalization.normalize(x=matError, n=5, t=6)
        aux = self.auxiliary.f(x=normalized, a = 10000)
        self.assertEqual(aux, 1)

    def test_UniformInitialization(self):
        initialize = InitializationStore.get('uniform')
        population = initialize.initialize(population_size = 100, 
                                                            n_concepts = 5)
        
        self.assertEqual(100, len(population))
    
    def test_PopulationEvaluation(self):
        initialize = InitializationStore.get('uniform')
        population = initialize.initialize(population_size = 100, 
                                                            n_concepts = 5)
        evaluation = PopulationEvaluation()
        init_state_d = self.data.iloc[0].to_dict()
        evaluated = evaluation.evaluate(population=population, data=self.data, l=5, transfer='sigmoid', inference='mKosko',
                                        normalization_type='L2', fitness_type='stach_fitness',
                                        a=10000, p=2)
        
