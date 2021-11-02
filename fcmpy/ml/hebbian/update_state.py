import numpy as np
from abc import ABC
from abc import abstractmethod
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.store.methodsStore import InferenceStore
from fcmpy.store.methodsStore import TransferStore


class UpdateStateVector(ABC):
    
    @abstractmethod
    def update(**kwargs) -> np.array:
        raise NotImplementedError('Update method is not defined!')


class FcmUpdate(UpdateStateVector):
    """
        Update the state vector based on the selected inference method.
    """
    @staticmethod
    def update(**kwargs) -> dict:
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
            ------
            y: dict
                keys -> concepts, values -> state values
        """
        weight_mat = kwargs['weight_matrix'].to_numpy()
        state_array = np.array(list(kwargs['state_vector'].values()))
        transfer_method = kwargs['transfer']
        inference_method = kwargs['inference']
        
        # get the methods for the simulation.
        inference = InferenceStore.get(inference_method)()
        transfer = TransferStore.get(transfer_method)()

        inferred = inference.infer(initial_state=state_array, weight_matrix=weight_mat, params=kwargs) # Inference
        state_vector = transfer.transfer(x=inferred, params=kwargs) # Apply transfer func on the results
        
        # convert to dict
        res = {i:y for i,y in zip(kwargs['weight_matrix'].columns, state_vector)}
        
        return res


class FcmUpdateAsynch(UpdateStateVector):
    """
        Asynchronous update of the FCM concept values.
    """
    @staticmethod
    @type_check
    def update(**kwargs) -> dict:
        """
            Asynchronously update the state vector.

            Parameters
            ----------
            source: str,
                        the activated concept

            state_vector: dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"            
            
            Return
            ------
            y: dict
                keys -> concepts, values -> state values
        """
        # extract the arguments
        source = kwargs['source']
        state = kwargs['state_vector'].copy()
        weight_matrix = kwargs['weight_matrix'].copy()

        # get the methods for the simulation.
        transfer_method = kwargs['transfer']
        transfer = TransferStore.get(transfer_method)()        
        
        for s in source:
            res = state[s] + weight_matrix[s].dot(list(state.values()))
            state[s] = transfer.transfer(x=res, params=kwargs)
        return state