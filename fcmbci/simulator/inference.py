import numpy as np
from simulator.transfer import Transfer

class Inference(Transfer):
    def __init__(self):
        super().__init__()
        self.inference_methods = {"kosko" : self.__kosko, "mKosko" : self.__mKosko, "rescaled" : self.__rescaled}

    @staticmethod
    def __mKosko(initial_state, weight_mat, **params):
        """
        Modified Kosko inference method.

        Parameters
        ----------
        initial_state: numpy.array
                        initial state vector of the concepts
        weight_mat: numpy.ndarray
                        N*N weight matrix of the FCM.

        Return
        ----------
        y: numpy.array
                updated state vector
        """
        res = weight_mat.dot(initial_state) + initial_state
        
        return res
    
    @staticmethod
    def __kosko(initial_state, weight_mat, **params):
        """
        Modified Kosko inference method.

        Parameters
        ----------
        initial_state: np.array
                            initial state vector of the concepts
        weight_mat: numpy.ndarray
                        N*N weight matrix of the FCM.
        
        Return
        ----------
        y: numpy.array
                updated state vector
        """
        res = weight_mat.dot(initial_state)
        
        return res
        
    @staticmethod   
    def __rescaled(initial_state, weight_mat, **params):
        """
        Rescaled inference method.

        Parameters
        ----------
        initial_state: numpy.array
                        initial state vector of the concepts
        weight_mat: numpy.ndarray
                        N*N weight matrix of the FCM.
        
        Return
        ----------
        y: numpy.array
                updated state vector
        """
        res = weight_mat.dot(([2*i-1 for i in initial_state]))+([2*i-1 for i in initial_state])
        
        return res
    
    def add_inference_method(self, func):
        """
        Add a new inference method.

        Parameters
        ----------
        func: dict
                key is the name of the method, value is the associated function
        """
        self.inference_methods.update(func)
    
    def remove_inference_method(self, func_name):

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