import numpy as np
from data_processor.input_validator import type_check

class FuzzyInference:

    """
    The class includes fuzzy inference methods for deriving FCM edge weights.

    Methods:
            __init__(self)
            __min(mf_x, weight, **params)
            __product(mf_x, weight, **params)
            add_fuzzy_inference_func(self, func)
            remove_fuzzy_inference_func(self, func_name)
    """

    def __init__(self):
        self.fuzzy_inference_funcs = {"min" : self.__min, "product" : self.__product}

    @staticmethod
    @type_check
    def __min(mf_x: np.ndarray, weight: float) -> np.ndarray:
        """
        Mamdani min fuzzy implication method.

        Parameters
        ----------
        mf_x: numpy.ndarray,
                membership function of a linguistic term (x)

        weight: float,
                    the weight at which the membership function x should be activated (i.e., the cut point)
        
        Return
        ---------
        y: numpy.ndarray
            the activated membership function
        """

        return np.fmin(weight, mf_x)

    @staticmethod
    @type_check
    def __product(mf_x: np.ndarray, weight: float) -> np.ndarray:
        """
        Product fuzzy implication method

        mf_x: numpy.ndarray.
                membership function of a linguistic term (x)

        weight: float,
                    the weight at which the membership function x should be activated (i.e., rescaled)
        
        Return
        ---------
        y: numpy.ndarray
            the activated (rescaled) membership function
        """

        return  np.dot(mf_x, weight)
    
    @type_check
    def add_fuzzy_inference_func(self, func: dict):
    
        """
        Add a fuzzy inference function.

        Parameters
        ----------
        func: dict,
                key is the name of the function, value is the associated function.
        """

        self.fuzzy_inference_funcs.update(func)
    
    @type_check
    def remove_fuzzy_inference_func(self, func_name: str):
        """
        Remove a fuzzy inference function.

        Parameters
        ----------
        func_name: str
                    name of the function to be removed.
        """
        if 'FuzzyInference.__' not in str(self.fuzzy_inference_funcs[func_name]):
            del self.fuzzy_inference_funcs[func_name]
        else:
            raise ValueError('Cannot remove a base function!')
