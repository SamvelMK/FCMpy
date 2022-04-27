import numpy as np
from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import GradientStore

class Update(ABC):
    """
        Interface for updating the parameter matrix W.
    """
    @abstractmethod
    def calculate(**kwargs):
        raise NotImplementedError('calculate method is not defined.')


class DeltaW(Update):
    """
        Calculate the change of the parameter matrix W.
    """
    @staticmethod
    def calculate(data:np.array, simulated, state_vector, weight_matrix, transfer:str, inference:str, **kwargs):
        """
            Calculate the change of the parameter matrix W.
        """
        dx_squared = GradientStore.get(method='dxSquared').compute
        dx_inference = GradientStore.get(method=inference).compute
        dx_transfer = GradientStore.get(method=transfer).compute

        dx_error = dx_squared(observed=data, simulated=simulated)
        dx_infer = dx_inference(state_vector=state_vector)
        dx_trans = dx_transfer(x=state_vector, weight_matrix = weight_matrix, inference=inference, params=kwargs)
    
        return (dx_error * dx_trans * dx_infer)*np.abs(np.sign(weight_matrix))