from matplotlib.font_manager import _Weight
import numpy as np
from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import InferenceStore
from fcmpy.store.methodsStore import TransferStore
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
    def calculate(inference:str, transfer:str, data:np.array, state_vector, weight_matrix, **kwargs):
        """
            Calculate the change of the parameter matrix W.
        """
        infer = InferenceStore.get(method=inference).infer
        trans = TransferStore.get(method=transfer).transfer
        dx_squared = GradientStore.get(method='dx_squared').compute
        dx_inference = GradientStore.get(method=inference).compute
        dx_transfer = GradientStore.get(method=transfer).compute

        infered  = infer(initial_state = state_vector, weight_matrix=weight_matrix)
        simulated = trans(x=infered, params=kwargs)

        dx_error = dx_squared(observed=data, simulated=simulated)
        dx_infer = dx_inference(state_vector=state_vector)
        dx_trans = dx_transfer(x=infered, params=kwargs)

        return (dx_error * dx_trans * dx_infer)*np.abs(np.sign(weight_matrix))