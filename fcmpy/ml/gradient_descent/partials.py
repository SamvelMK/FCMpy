from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import TransferStore


class Gradient(ABC):
    """
        Interface for computing gradients.
    """
    @abstractmethod
    def compute(**kwargs):
        raise NotImplementedError('compute method is not defined.')


class DxSquaredErrors(Gradient):
    """
        Partial derivative of the squared errors w.r.t. matrix W.
    """
    def compute(**kwargs):
        simulated = kwargs['simulated']
        observed = kwargs['observed']
        return 2*(observed - simulated)


class DxSigmoid(Gradient):
    """
        Partial derivative of the sigmoid transfer function w.r.t. matrix W.
    """

    @staticmethod
    def compute(**kwargs):
        """
            Compute the derivative of the sigmoid transfer function w.r.t. matrix W.

            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            l : int/float
                    A parameter that determines the steepness of the sigmoid function at values around 0. 
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0,1].
        """
        sigmoid = TransferStore.get('sigmoid').transfer
        x = kwargs['x']
        return sigmoid(x, params = kwargs)*(1-sigmoid(x, params = kwargs))


class DxTanh(Gradient):
    """
        Partial derivative of the tanh transfer function w.r.t. matrix W.
    """
    @staticmethod
    def compute(**kwargs):
        """
            Compute the derivative of the hyperbolic tangent transfer function w.r.t. matrix W.

            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0,1].
        """
        tanh = TransferStore.get('tanh').transfer
        x = kwargs['x']
        return tanh(x)*(1-tanh(x)**2)


class DxKosko(Gradient):
    """
        Partial derivative of the Kosko's (modified kosko's) inference method w.r.t. matrix W.
    """
    @staticmethod
    def compute(**kwargs):
        """
            Compute the partial derivative of the ksoko's (and modified kosko's) inference function w.r.t. matrix W.

            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0,1].
        """
        state_vector = kwargs['state_vector']
        return state_vector.reshape(len(state_vector), 1)


class DxRescaled(Gradient):
    """
        Partial derivative of the rescaled Kosko's (modified kosko's) inference method w.r.t. matrix W.
    """
    @staticmethod
    def compute(**kwargs):
        """
            Compute the partial derivative of the rescaled ksoko's inference function w.r.t. matrix W.

            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0,1].
        """
        state_vector = kwargs['state_vector']
        x = (2 * state_vector - 1)
        return x.reshape(len(x), 1)