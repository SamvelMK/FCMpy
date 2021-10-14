import numpy as np
from abc import ABC, abstractmethod


class Inference(ABC):
    """
        Class of FCM inference methods
    """
    @abstractmethod
    def infer() -> np.array:
        raise NotImplementedError('Infer method is not defined!')


class Kosko(Inference):
    """
        Kosko's inference method.
    """
    @staticmethod    
    def infer(**kwargs) -> np.array:
        """
            Kosko's inference method.

            Parameters
            ----------
            initial_state: numpy.array
                                initial state vector of the concepts
            weight_matrix: numpy.ndarray
                            N*N weight matrix of the FCM.
            
            Return
            -------
            y: numpy.array
                    updated state vector
        """
        initial_state = kwargs['initial_state']
        weight_matrix = kwargs['weight_matrix']

        weight_matrix = weight_matrix.T
        res = weight_matrix.dot(initial_state)
        
        return res


class ModifiedKosko(Inference):
    """
        Modified Kosko inference method.
    """
    @staticmethod    
    def infer(**kwargs) -> np.array:
        """
            Modified Kosko inference method.

            Parameters
            ----------
            initial_state: numpy.array
                            initial state vector of the concepts
            weight_matrix: numpy.ndarray
                            N*N weight matrix of the FCM.

            Return
            ----------
            y: numpy.array
                    updated state vector
        """
        initial_state = kwargs['initial_state']
        weight_matrix = kwargs['weight_matrix']

        weight_matrix = weight_matrix.T
        res = weight_matrix.dot(initial_state) + initial_state
        
        return res


class Rescaled(Inference):
    """
        Rescaled inference method.
    """
    @staticmethod    
    def infer(**kwargs) -> np.array:
        """
            Rescaled inference method.

            Parameters
            ----------
            initial_state: numpy.array
                            initial state vector of the concepts
            weight_matrix: numpy.ndarray
                            N*N weight matrix of the FCM.
            
            Return
            ----------
            y: numpy.array
                    updated state vector
        """
        initial_state = kwargs['initial_state']
        weight_matrix = kwargs['weight_matrix']
        weight_matrix = weight_matrix.T

        res = weight_matrix.dot(([2*i-1 for i in initial_state])) + ([2*i-1 for i in initial_state])
        
        return res