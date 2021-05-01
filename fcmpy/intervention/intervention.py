import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pandas as pd
import numpy as np
from fcmpy.simulator.simulator import FcmSimulator
import warnings
from fcmpy.expert_fcm.input_validator import type_check
from typing import Union
from abc import ABC, abstractmethod

class Intervention(ABC):
    
    """
    Test intervention scenarios.
    """
    
    @abstractmethod
    def add_intervention():
        raise NotImplementedError('add_intervention method is not defined!')

    @abstractmethod
    def remove_intervention():
        raise NotImplementedError('remove_intervention method is not defined!')

    @abstractmethod
    def test_intervention():
        raise NotImplementedError('test_intervention method is not defined!')

class FcmIntervention(Intervention):

    """
    The class includes methods for testing interventions (what-if scenarios) on top of a defined FCM structure.

    Methods:
        __init__(self, initial_state, weight_matrix, transfer, inference, thresh, iterations, **params)
        add_intervention(self, name, weights, effectiveness)
        remove_intervention(self, name)
        test_intervention(self, name, iterations = None)
    """
    
    def __init__(self, simulator):
        
        """
        Parameters
        ----------
        simulator: Simulator
        """

        self.__simulator = simulator()
        self.__interventions = {}
        self.__test_results = {}
        self.__equilibriums = {}
        self.__comparison_table = None

    @property
    def test_results(self):
        return self.__test_results
    
    @property
    def interventions(self):
        return self.__interventions
    
    @property
    def equilibriums(self):
        return pd.DataFrame(self.__equilibriums)
    
    @property
    def comparison_table(self):
        diff = {}
        df = pd.DataFrame(self.__equilibriums)
        for i in df.columns:
            diff[i] = ((df[i] - df.iloc[:, 0])/df.iloc[:, 0])*100
        self.__comparison_table = pd.DataFrame(diff)
        return self.__comparison_table

    @type_check
    def initialize(self, initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], 
                            transfer: str, inference: str, thresh: float, iterations: int, l=None, **params):
        
        """
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

        l: None
            A parameter that determines the steepness of the sigmoid function at values around 0.

        **params: additional parameters
        """

        self.__weight_matrix = weight_matrix
        self.__initial_state=initial_state
        self.__transfer = transfer
        self.__inference = inference
        self.__thresh = thresh
        self.__iterations = iterations
        self.__l = l
        
        self.__test_results['baseline'] = self.__simulator.simulate(initial_state = self.__initial_state, weight_matrix = self.__weight_matrix,
                                                                transfer = self.__transfer, inference = self.__inference, thresh = self.__thresh, 
                                                                iterations = self.__iterations, l=self.__l)
        
        self.__equilibriums['baseline'] = self.test_results['baseline'].iloc[-1]

    @type_check
    def add_intervention(self, name: str, impact: dict, effectiveness: Union[int, float]):

        """
        Add an intervention node with the associated causal weights to the FCM.

        Parameters
        ----------
        name: str
                name of the intervention

        impact: dict
                    keys ---> concepts the intervention impacts, value: the associated causal weight

        effectiveness: float
                        the degree to which the intervention was delivered (should be between [-1, 1])
        """

        # Check whether the passed intervention inputs are in the functions' domain.
        if (min(list(impact.values())) < -1) or (max(list(impact.values())) > 1):
            raise ValueError('the values in the causal weights are out of the domain [-1,1].')
        elif (effectiveness < 0) or (effectiveness > 1):
            raise ValueError('the values in the intervention effectiveness are out of the domain [0,1].')

        intervention = {}
        intervention['efectiveness'] = effectiveness
        
        # construct a weight matrix for a given intervention
        if type(self.__weight_matrix) == np.ndarray:
            temp = pd.DataFrame(self.__weight_matrix, columns=self.__initial_state)
        else:
            temp = self.__weight_matrix.copy(deep=True)

        temp['antecident'] = temp.columns
        temp.set_index('antecident', inplace=True)
        temp['intervention'] = 0
        temp.loc[len(temp)] = 0
        temp.rename(index = {temp.index[-1] : 'intervention'}, inplace = True)
        
        # add the intervention impact
        for key in impact.keys():
            temp.loc['intervention', key] = impact[key]
            
        # construct the new state vector for a given intervention (baseline + intervention effectiveness)
        temp_vector = self.__equilibriums['baseline'].copy(deep=True)
        temp_vector = temp_vector.append(pd.Series({'intervention': effectiveness})).to_dict()
        
        # add the causal weights for the intervention
        intervention['weight_matrix'] = temp
        intervention['state_vector'] = temp_vector
        self.__interventions[name] = intervention    
    
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

    @type_check
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

        weight_matrix = self.__interventions[name]['weight_matrix']
        state_vector = self.__interventions[name]['state_vector']
        
        self.__test_results[name] = self.__simulator.simulate(initial_state=state_vector, weight_matrix=weight_matrix, transfer=self.__transfer, 
                                                                inference=self.__inference, thresh=self.__thresh, 
                                                                iterations=iterations, l = self.__l)
        
        self.__equilibriums[name] = self.__test_results[name].iloc[-1][:-1]