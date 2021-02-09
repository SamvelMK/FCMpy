import numpy as np
from simulator.transfer import Transfer
from data_processor.input_validator import type_check

class Inference(Transfer):
    """
    The class includes inference methods for FCM update function.

    Methods:
            __init__(self)
            __mKosko(initial_state, weight_matrix, **params)
            __kosko(initial_state, weight_matrix, **params)
            __rescaled(initial_state, weight_matrix, **params)
            add_inference_methods(self, func)
            remove_inference_methods(self, func_name)
    """
    def __init__(self):
        super().__init__()
        self.inference_methods = {"kosko" : self.__kosko, "mKosko" : self.__mKosko, "rescaled" : self.__rescaled}
           
    @staticmethod
    @type_check
    def __mKosko(initial_state: np.ndarray, weight_matrix: np.ndarray, **params) -> np.ndarray:

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

        weight_matrix = weight_matrix.T
        res = weight_matrix.dot(initial_state) + initial_state
        
        return res
    
    @staticmethod
    @type_check
    def __kosko(initial_state: np.ndarray, weight_matrix: np.ndarray, **params) -> np.ndarray:

        """
        Modified Kosko inference method.

        Parameters
        ----------
        initial_state:  numpy.array
                            initial state vector of the concepts
        weight_matrix: numpy.ndarray
                        N*N weight matrix of the FCM.
        
        Return
        ----------
        y: numpy.array
                updated state vector
        """

        weight_matrix = weight_matrix.T
        res = weight_matrix.dot(initial_state)
        
        return res
        
    @staticmethod
    @type_check   
    def __rescaled(initial_state: np.ndarray, weight_matrix: np.ndarray, **params) -> np.ndarray:

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

        weight_matrix = weight_matrix.T

        res = weight_matrix.dot(([2*i-1 for i in initial_state]))+([2*i-1 for i in initial_state])
        
        return res
    
    @type_check
    def add_inference_method(self, func: dict):
        """
        Add a new inference method.

        Parameters
        ----------
        func: dict
                key is the name of the method, value is the associated function
        """
        self.inference_methods.update(func)

    @type_check
    def remove_inference_method(self, func_name: str):

        """
        Remove an inference method.

        Parameters
        ----------
        key: str
                name of the method to be removed
        """
        
        if 'Inference.__' not in str(self.inference_methods[func_name]):
            del self.inference_methods[func_name]
        else:
            raise ValueError('Cannot remove a base function!')