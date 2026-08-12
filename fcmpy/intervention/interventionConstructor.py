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
            initial_state: dict
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
    """
        Construct continuous interventions: a persistent driver node is added to the
        FCM structure and applied on top of the converged baseline equilibrium.
    """
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
                            the degree to which the intervention was delivered (should be between [0, 1])
                            default --> 1
                            
            Return
            ----------
            y: dictionary
        """
        weight_matrix = kwargs['weight_matrix']
        initial_state = kwargs['initial_state']
        equilibriums = kwargs['equilibriums']

        try:
            impact = kwargs['params']['impact']
        except KeyError:
            raise ValueError("Continuous interventions require an 'impact' dict, "
                                "e.g. impact={'C1': -0.3}.")

        # Set the intervention effectiveness to 1 if the optional parameter is not specified.
        try:
            effectiveness = kwargs['params']['effectiveness']
        except KeyError:
            effectiveness = 1

        # Check whether the passed intervention inputs are in the function's domain.
        impact_values = list(impact.values())
        if (min(impact_values) < -1) or (max(impact_values) > 1):
            raise ValueError('the values in the causal weights are out of the domain [-1,1].')
        elif (effectiveness < 0) or (effectiveness > 1):
            raise ValueError('the values in the intervention effectiveness are out of the domain [0,1].')

        intervention = {}
        intervention['effectiveness'] = effectiveness
        
        # construct a weight matrix for a given intervention
        if type(weight_matrix) == np.ndarray:
            # an ndarray carries no labels at all, so columns/index must both
            # come from initial_state, in the same order, by construction.
            temp = pd.DataFrame(weight_matrix, columns=list(initial_state), index=list(initial_state))
        else:
            temp = weight_matrix.copy(deep=True)
            if set(temp.index) != set(temp.columns):
                # no meaningful row labels (e.g. the default RangeIndex produced by
                # pd.DataFrame([...], columns=[...])) -- fall back to the documented
                # convention that row i corresponds to columns[i].
                temp.index = temp.columns
            # else: the index already labels each row correctly (whatever its
            # physical order relative to columns) -- trust it as-is instead of
            # overwriting it positionally, which would silently misattribute edges.

        temp['intervention'] = 0.0
        temp.loc['intervention'] = 0.0
        
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