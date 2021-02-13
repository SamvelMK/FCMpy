import pandas as pd
import numpy as np
from simulator.inference import Inference
from simulator.simulator import Simulator
import warnings
from data_processor.checkers import Checker
from data_processor.input_validator import type_check
from typing import Union

class Intervention(Simulator):

    """
    The class includes methods for testing interventiosn (what-if scenarios) on top of a defined FCM structure.

    Methods:
            __init__(self, initial_state, weight_matrix, transfer, inference, thresh, iterations, **params)
            add_intervention(self, name, weights, effectiveness)
            remove_intervention(self, name)
            test_intervention(self, name, iterations = None)
    """

    @type_check
    def __init__(self, initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], transfer: str, inference: str, thresh: float, iterations: int, **params):

        """
        Test interventions (simulate what-if scenarios) on top of a defined FCM structure.

        Parameters
        ----------
        initial_state: dict
                        keys ---> concepts, values ---> initial states of the associated concepts

        weight_matrix: panda.DataFrame
                    causal weights between concepts

        transfer: str
                    transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

        inference: str
                    inference method --> "kosko", "mKosko", "rescaled"

        thresh: float
                    threshold for the error

        iterations: int
                        number of iterations

        params: additional parameters for the methods
        """

        Checker.input_check(initial_state=initial_state, weight_matrix= weight_matrix)

        super().__init__()
        self.interventions = {}
        self.test_results = {}
        self.weight_matrix = weight_matrix
        self.__initial_state=initial_state
        self.__transfer = transfer
        self.__inference = inference
        self.__thresh = thresh
        self.__iterations = iterations
        self.__params = params

        self.test_results['baseline'] = self.simulate(initial_state = self.__initial_state, weight_matrix = self.weight_matrix,
                                                   transfer = self.__transfer, inference = self.__inference, thresh = self.__thresh, 
                                                   iterations = self.__iterations, **self.__params)
        
        self.initial_equilibrium = self.test_results['baseline'].iloc[-1]

    @type_check
    def add_intervention(self, name: str, weights: dict, effectiveness: Union[int, float]):

        """
        Add an intervention node with the associated causal weights to the FCM.

        Parameters
        ----------
        name: str
                name of the intervention

        weights: dict
                    keys ---> concepts the intervention impacts, value: the associated causal weight

        effectiveness: float
                        the degree to which the intervention was delivered (should be between [-1, 1])
        """

        # Check whether the passed intervention inputs are in the functions' domain.
        if (min(list(weights.values())) < -1) or (max(list(weights.values())) > 1):
            raise ValueError('the values in the causal weights are out of the domain [-1,1].')
        elif (effectiveness < 0) or (effectiveness > 1):
            raise ValueError('the values in the intervention effectiveness are out of the domain [0,1].')

        intervention = {}
        intervention['efectiveness'] = effectiveness
        

        # construct a weight matrix for a given intervention
        if type(self.weight_matrix) == np.ndarray:
            temp = pd.DataFrame(self.weight_matrix, columns=self.__initial_state)
        else:
            temp = self.weight_matrix.copy(deep=True)
        temp['antecident'] = temp.columns
        temp.set_index('antecident', inplace=True)
        temp['intervention'] = 0
        temp.loc[len(temp)] = 0
        temp.rename(index = {temp.index[-1] : 'intervention'}, inplace = True)
        
        # add the intervention impact
        for key in weights.keys():
            temp.loc['intervention', key] = weights[key]
            
        # construct the new state vector for a given intervention (baseline + intervention effectiveness)
        temp_vector = self.initial_equilibrium.copy(deep=True)
        temp_vector = temp_vector.append(pd.Series({'intervention': effectiveness})).to_dict()
        
        # add the causal weights for the intervention
        intervention['weight_matrix'] = temp
        intervention['state_vector'] = temp_vector
        self.interventions[name] = intervention    
    
    @type_check
    def remove_intervention(self, name: str):

        """
        Remove intervention.

        Parameters
        ----------
        name: str
                name of the intervention
        """

        del self.interventions[name]

    def test_intervention(self, name: str, iterations: int = None):
        
        """
        Test an intervention case.

        Parameters
        ----------
        name: str
                name of the intervention
                
        iterations: number of iterations for the FCM simulation
                        default ---> the iterations specified in the init.
        """
        
        if iterations:
            iterations = iterations
        else:
            iterations = self.__iterations

        weight_matrix = self.interventions[name]['weight_matrix']
        state_vector = self.interventions[name]['state_vector']
        
        self.test_results[name] = self.simulate(initial_state=state_vector, weight_matrix=weight_matrix, transfer=self.__transfer, 
                                                                inference=self.__inference, thresh=self.__thresh, 
                                                                iterations=iterations, **self.__params)