import pandas as pd
import numpy as np
import warnings
from abc import ABC
from abc import abstractmethod
from typing import Union
from tqdm import tqdm
from fcmpy.expert_fcm.input_validator import type_check
from .update_weight import NhlWeightUpdate
from .update_state import FcmUpdate
from .termination import FirstCriterion
from .termination import SecondCriterion


class HebbianLearning(ABC):
    """
        Hebbian Based Learning for FCMs.
    """
    @abstractmethod
    def run(**kwargs) -> pd.DataFrame:
        raise NotImplementedError('run method is not defined.')


class NHL(HebbianLearning):
    """
        NHL algorithm for optimizing the FCM weight matrix.
    """
    @type_check
    def __init__(self, state_vector: dict, weight_matrix:Union[np.ndarray, pd.DataFrame], 
                    doc_values:dict) -> None:
        """
            NHL algorithm for optimizing the FCM weight matrix.

            Parameters
            ----------
            state_vector : dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept
            
            weight_matrix : pd.DataFrame
                            N*N weight matrix of the FCM. 
            
            doc_values : dict
                            DOC values where the keys are the DOCs and 
                            the values are lists with min max values of the DOCs.
        """
        self.__state_vector = state_vector
        self.__weight_matrix = weight_matrix
        self.__doc_values = doc_values
        self.__weightUpdate = NhlWeightUpdate()
        self.__stateUpdate = FcmUpdate()
        self.__termination1 = FirstCriterion()
        self.__termination2 = SecondCriterion()

    @type_check
    def run(self, decay=1, learning_rate=0.01, iterations:int= 100, transfer:str= 'sigmoid', 
                    inference:str='mKosko', thresh:float = 0.002, l:Union[float, int]=0.98, **kwargs) -> pd.DataFrame:
        """
            Run the NHL algorithm.

            Parameters
            ----------
            decay: float,
                    default --> 1
            
            learning_rate: int/float
                            default --> 0.01
            
            iterations: int
                        iterations for the NHL algorithm to run
                        default --> 100

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
                        default -> "sigmoid"

            l: float
                parameter for the sigmoid transfer function

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"

            thresh: float
                        threshold of error for the F2 termination condition for NHL algorithm
                        default --> 0.002

            Return
            -------
            y : pd.DataFrame
                    the optimized weight matrix
        """
        gamma = decay
        eta = learning_rate
        # Initialize the prior state vector and weight matrices.
        s_prior = self.__state_vector.copy()
        w_prior = self.__weight_matrix.copy()

        for _ in tqdm(range(iterations)):
            w_new = self.__weightUpdate.update(initial_state=s_prior, weight_matrix=w_prior,
                                                    gamma=gamma, eta=eta)

            s_new = self.__stateUpdate.update(state_vector=s_prior, weight_matrix=w_new, transfer=transfer,
                                                inference=inference, l=l, kwargs=kwargs) 
            
            if self.__termination1.terminate(doc_values=self.__doc_values, state_vector_prior=s_prior, state_vector_current=s_new) and \
                                                self.__termination2.terminate(doc_values=self.__doc_values, state_vector_prior=s_prior, 
                                                                    state_vector_current=s_new, thresh=thresh):

                print(f'The NHL learning process converged at step {_} with the learning rate eta = {eta} and decay = {gamma}!')
                return w_new
            else:
                s_prior = s_new
                w_prior = w_new
        
        if _ >= iterations-1:
            warnings.warn('The NHL did not converge! Consider a different set of parameters.')
            return w_new