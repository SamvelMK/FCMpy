from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import InferenceStore, TransferStore
import numpy as np

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
    def compute(**kwargs) -> np.array:
        """
            Compute the partial derivative of the mse w.r.t matrix W.
            
            Parameters
            ----------
            predicted: np.array
                        predicted data
            
            observed: np.array
                        observed data
            
            Return
            ------
            y: np.array
                partial derivatives of the MSE w.r.t matrix W
        """
        predicted = kwargs['predicted']
        observed = kwargs['observed']
        return -2*(observed-predicted)


class DxSigmoid(Gradient):
    """
        Partial derivative of the sigmoid transfer function w.r.t. matrix W.
    """

    @staticmethod
    def compute(**kwargs):
        """
            Compute the partial derivative of the sigmoid transfer function w.r.t. matrix W.

            Parameters
            ----------
            x : np.array
                predicted data

            weight_matrix: np.array
                            connection matrix at time t
            l : int/float
                    A parameter that determines the steepness of the sigmoid function at values around 0. 
            inference: str
                    inference method --> "kosko", "mKosko", "rescaled"
            Return
            -------
            y : np.array
                partial derivative of the sigmoid function w.r.t matrix W.
        """
        sigmoid = TransferStore.get('sigmoid').transfer
        state_vector = kwargs['x']
        weight_matrix = kwargs['weight_matrix']
        inference = kwargs['inference']
        l = kwargs['params']['l']
        infer = InferenceStore.get(inference).infer
        x = infer(initial_state=state_vector, weight_matrix=weight_matrix)
        return sigmoid(x=x, params = {'l':l})*(1-sigmoid(x=x,  params = {'l':l}))


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
            x : np.array
                predicted data

            weight_matrix: np.array
                            connection matrix at time t
            l : int/float
                    A parameter that determines the steepness of the sigmoid function at values around 0. 
            inference: str
                    inference method --> "kosko", "mKosko", "rescaled"
            Return
            -------
            y : np.array
                partial derivative of the hyperbolic tangent w.r.t matrix W.
        """
        tanh = TransferStore.get('tanh').transfer
        state_vector = kwargs['x']
        weight_matrix = kwargs['weight_matrix']
        inference = kwargs['inference']
        infer = InferenceStore.get(inference).infer
        x = infer(initial_state=state_vector, weight_matrix=weight_matrix)
        return (1-tanh(x=x)**2)


class DxKosko(Gradient):
    """
        Partial derivative of the Kosko's (and modified kosko's) inference method w.r.t. matrix W.
    """
    @staticmethod
    def compute(**kwargs):
        """
            Compute the partial derivative of the ksoko's (and modified kosko's) inference function w.r.t. matrix W.

            Parameters
            ----------
            state_vector : np.array
                            state vector at time t

            Return
            -------
            y : np.array
                partial derivative Kosko's and Modified kosko's inference function w.r.t matrix W.
        """
        state_vector = kwargs['state_vector']
        return state_vector.reshape(len(state_vector), 1)


class DxRescaled(Gradient):
    """
        Partial derivative of the rescaled inference method w.r.t. matrix W.
    """
    @staticmethod
    def compute(**kwargs):
        """
            Compute the partial derivative of the rescaled inference function w.r.t. matrix W.

            Parameters
            ----------
            state_vector : np.array
                            state vector at time t

            Return
            -------
            y : np.array
                partial derivative rescaled inference function w.r.t matrix W.
        """
        state_vector = kwargs['state_vector']
        x = (2 * state_vector - 1)
        return x.reshape(len(x), 1)