import pandas as pd
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import InterventionStore
from fcmpy.expert_fcm.input_validator import type_check


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
            initialize(self, initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], 
                                transfer: str, inference: str, thresh: float, iterations: int, l=1, 
                                output_concepts = None, convergence = 'absDiff',  **params)

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
        # mult 100 to get the percentages directly
        df = pd.DataFrame(self.__equilibriums)*100
        for i in df.columns:
            diff[i] =  df[i] - df.iloc[:, 0]
        self.__comparison_table = pd.DataFrame(diff)
        return self.__comparison_table

    @type_check
    def initialize(self, initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], 
                            transfer: str, inference: str, thresh: float, iterations: int, l=1, 
                            output_concepts = None, convergence = 'absDiff',  **params):
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

            l: 1
                A parameter that determines the steepness of the sigmoid function at values around 0.
            
            output_concepts: bool, list
                                the output concepts for the convergence check
                                default --> None
            
            convergence: str,
                            convergence method
                            default --> 'absDiff': absolute difference between the simulation steps

            **params: additional parameters
        """
        self.__weight_matrix = weight_matrix
        self.__initial_state=initial_state
        self.__transfer = transfer
        self.__inference = inference
        self.__thresh = thresh
        self.__iterations = iterations
        self.__l = l
        self.__output_concepts = output_concepts
        self.__convergence = convergence
        
        self.__test_results['baseline'] = self.__simulator.simulate(initial_state = self.__initial_state, weight_matrix = self.__weight_matrix,
                                                                transfer = self.__transfer, inference = self.__inference, thresh = self.__thresh, 
                                                                iterations = self.__iterations, l=self.__l, output_concepts = self.__output_concepts,
                                                                convergence = self.__convergence, params = params)
        
        self.__equilibriums['baseline'] = self.test_results['baseline'].iloc[-1]

    @type_check
    def add_intervention(self, name, type='continuous', **kwargs):
        """
            Add an intervention node with the associated causal weights to the FCM.

            Parameters
            ----------
            name: str
                    name of the intervention
            
            type: str
                    type of intervention
                    default --> continuous

            impact: dict
                        keys --> concepts the intervention impacts, value: the associated causal weight

            effectiveness: float
                            the degree to which the intervention was delivered (should be between [0, 1])
                            default --> 1
        """
        if type != 'continuous':
            s = self.__initial_state.copy()
            s.update(kwargs['initial_state'])
            initial_state = s.copy()
        else:
            initial_state = self.__initial_state

        constructor = InterventionStore.get(type)()
        self.__interventions[name] = constructor.build(weight_matrix=self.__weight_matrix, initial_state=initial_state,
                                                        equilibriums = self.__equilibriums, params=kwargs)    
    
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
        
        self.__test_results[name] = self.__simulator.simulate(initial_state=state_vector, weight_matrix=weight_matrix, 
                                                                transfer=self.__transfer, inference=self.__inference, 
                                                                thresh=self.__thresh, iterations=iterations,
                                                                l = self.__l, output_concepts=self.__output_concepts,
                                                                convergence=self.__convergence)
        
        self.__equilibriums[name] = self.__test_results[name].iloc[-1][:len(self.__initial_state)]