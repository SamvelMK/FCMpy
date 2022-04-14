import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class InterventionConstructor(ABC):
    """
        Class of methods for building interventions.
    """
    @abstractmethod
    def build() -> dict:
        raise NotImplementedError('Build method is not defined!')


class SingleShot(InterventionConstructor):
    """
        Construct single-shot interventions
    """ 
    @staticmethod
    def build(**kwargs) -> dict:
        """
            Construct a single shot intervention case

            Parameters
            ----------
            state_vector: dict
                            keys --> concepts, values --> state of the concept.

            weight_matrix: pd.DataFrame
                            causal weights between concepts

            Return
            ----------
            y: dictionary
        """
        intervention = {}
        intervention['state_vector'] = kwargs['initial_state']
        intervention['weight_matrix'] = kwargs['weight_matrix']
        
        return intervention


class Continuous(InterventionConstructor):
    
    @staticmethod
    def build(**kwargs) -> dict:
        """
            Construct a continuous intervention case

            Parameters
            ----------
            weight_matrix: pd.DataFrame
                            causal weights between concepts

            initial_state: dict
                            keys --> concepts, values --> state of the concept.

            equilibriums: dict
                            keys --> concepts, values --> equilibrium states of the concept.

            impact: dict
                        keys ---> concepts the intervention impacts, value: the associated causal weight

            effectiveness: float
                            the degree to which the intervention was delivered (should be between [-1, 1])
                            default --> 1
                            
            Return
            ----------
            y: dictionary
        """
        weight_matrix = kwargs['weight_matrix']
        initial_state = kwargs['initial_state']
        equilibriums = kwargs['equilibriums']
        impact = kwargs['params']['impact']
        
        # Set the intervention effectiveness to 1 if the optional parameter is not specified.
        try:
            effectiveness = kwargs['params']['effectiveness']
        except:
            effectiveness = 1

        # Check whether the passed intervention inputs are in the function's domain.
        if (min(list(impact.values())) < -1) or (max(list(impact.values())) > 1):
            raise ValueError('the values in the causal weights are out of the domain [-1,1].')
        elif (effectiveness < 0) or (effectiveness > 1):
            raise ValueError('the values in the intervention effectiveness are out of the domain [0,1].')

        intervention = {}
        intervention['effectiveness'] = effectiveness
        
        # construct a weight matrix for a given intervention
        if type(weight_matrix) == np.ndarray:
            temp = pd.DataFrame(weight_matrix, columns=initial_state)
        else:
            temp = weight_matrix.copy(deep=True)

        temp['antecedent'] = temp.columns
        temp.set_index('antecedent', inplace=True)
        temp['intervention'] = 0
        temp.loc[len(temp)] = 0
        temp.rename(index = {temp.index[-1] : 'intervention'}, inplace = True)
        
        # add the intervention impact
        for key in impact.keys():
            temp.loc['intervention', key] = impact[key]
            
        # construct the new state vector for a given intervention (baseline + intervention effectiveness)
        temp_vector = equilibriums['baseline'].copy(deep=True)
        temp_vector = pd.concat([temp_vector, pd.Series({'intervention': effectiveness})]).to_dict()
        
        # add the causal weights for the intervention
        intervention['weight_matrix'] = temp
        intervention['state_vector'] = temp_vector

        return intervention