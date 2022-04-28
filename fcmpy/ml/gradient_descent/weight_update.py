from abc import ABC, abstractmethod
import numpy as np

class WeightUpdate(ABC):
    """
        Interface for computing the change in the parameter matrix W.
    """
    @abstractmethod
    def update(**kwargs):
        raise NotImplementedError('update method is not defined.')


class VanillaGd(WeightUpdate):
    """
        Parameter updates for vanilla gradient descent.
    """
    @staticmethod
    def update(**kwargs):
        """
            The update function for vanilla gradient descent.

            Parameters
            ----------
            learning_rate:float
                            learning rate

            delta_w: np.array
                        the gradients
            
            Return
            ------
            y: np.array
                the change of the W during learning step t+1
        """
        delta_w = kwargs['delta_w']
        learning_rate = kwargs['learning_rate']
        return -learning_rate*delta_w


class Adam(WeightUpdate):
    """
        Parameter updates for Adam solver.
    """
    @staticmethod
    def update(**kwargs):
        """
            The update function for adam solver.

            Parameters
            ----------
            learning_rate:float
                            learning rate

            delta_w: np.array
                        the gradients
            
            Return
            ------
            y: np.array
                the change of the W during learning step t+1
        """
        b1 = kwargs['b1']
        b2 = kwargs['b2']
        mt = 0
        vt=0
        delta_w = kwargs['delta_w']
        epoch = kwargs['epoch']
        learning_rate = kwargs['learning_rate']
        e = kwargs['e']
        
        mt = b1*mt+(1-b1)*delta_w
        vt = b2*vt+(1-b2)*(delta_w)**2
        mthat = mt/(1-b1**(epoch+1))
        vthat = vt/(1-b2**(epoch+1))
        change = -(learning_rate/(np.sqrt(vthat)+e))*mthat

        return change