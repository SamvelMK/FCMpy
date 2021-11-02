import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod

class WeightUpdate(ABC):
    """
        Interface for updating the FCM weight matrix
    """
    @abstractmethod
    def update(**kwargs) -> np.ndarray:
        """
            Update the FCM weight matrix
        """
        raise NotImplementedError('Update method is not defined!')


class NhlWeightUpdate(WeightUpdate):
    """
        Synchronously (i.e., NHL) update the weight matrix according 
        to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(**kwargs) -> pd.DataFrame:
        """
            Update the weight matrix according to the adjusted Hebbian rule.

            Parameters
            ----------
            initial_state: dict
                            initial state vector of the concepts

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM
            
            gamma: float, int
                    decay coefficient
            
            eta: float,
                    learning rate
        
            Return
            ----------
            y: pd.DataFrame
                updated weight matrix
        """
        # Extract the arguments
        initial_state = kwargs['initial_state'].copy()
        weight_matrix = kwargs['weight_matrix'].copy()
        gamma = kwargs['gamma']
        eta = kwargs['eta']

        # convert pd -> ndarray
        # convert dict val -> array
        w = weight_matrix.to_numpy()
        s = np.array(list(initial_state.values()))

        # update the W matrix
        a = gamma*w
        b = eta*s
        c = abs(w)*s
        res = a + abs(np.sign(w))*b*(s-c.T).T
        
        # convert it back to df
        res = pd.DataFrame(res, columns = weight_matrix.columns, index = weight_matrix.columns)
        return res


class AhlWeightUpdate(WeightUpdate):
    """
        Asynchronous (i.e., AHL) Update the weight matrix according to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(**kwargs) -> float:
        """
            Asynchronously update the weight matrix according to the adjusted Hebbian rule.

            Parameters
            ----------
            source: str,
                        the activated concept

            state_vector: dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM.

            gamma: float, int
                    decay coefficient
            
            eta: float,
                    learning rate           
            
            Return
            ------
            y: pd.DataFrame
                updated weight matrix
        """
        # Extract arguments
        source = kwargs['source']
        state_vector = kwargs['state_vector']
        weight_matrix = kwargs['weight_matrix']
        gamma = kwargs['gamma']
        eta = kwargs['eta']

        gamma = 1-gamma
        w = weight_matrix.copy()
        for j in source:
            for i in weight_matrix:
                if j != i:
                    w.loc[j, i] = gamma*w.loc[j, i]+eta*state_vector[j]*(state_vector[i]-w.loc[j, i]*state_vector[j])
        return w
        