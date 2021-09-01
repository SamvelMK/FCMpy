from fcmpy.expert_fcm.input_validator import type_check
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
from fcmpy.store.methodsStore import InferenceStore
from fcmpy.store.methodsStore import TransferStore
from fcmpy.expert_fcm.input_validator import type_check

class UpdateStateVector(ABC):
    
    @abstractmethod
    def update(init_state: np.array, weight_matrix: np.ndarray, transfer:str, inference:str, **kwargs) -> np.array:
        raise NotImplementedError('Update method is not defined!')


class FcmUpdate(UpdateStateVector):
    """
        Update the state vector based on the selected inference method.
    """
    @staticmethod
    @type_check
    def update(state_vector:dict, weight_matrix:pd.DataFrame, transfer:str, inference:str, **kwargs) -> dict:
        """
            Update the state vector according to the selected inference method.

            Parameters
            ----------
            state_vector: dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"
            
            Return
            ----------
            y: dict
                keys -> concepts, values -> state values
        """
        # get the methods for the simulation.
        weight_mat = weight_matrix.to_numpy()
        state_array = np.array(list(state_vector.values()))
        transfer = TransferStore.get(transfer)()
        inference = InferenceStore.get(inference)()

        infered = inference.infer(initial_state=state_array, weight_matrix=weight_mat, params=kwargs) # Inference
        state_vector = transfer.transfer(x=infered, params=kwargs) # Apply transfer func on the results
        
        # convert to dict
        res = {i:y for i,y in zip(weight_matrix.columns, state_vector)}
        
        return res