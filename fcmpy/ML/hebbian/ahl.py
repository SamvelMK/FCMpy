import pandas as pd
import warnings
from typing import Union
from tqdm import tqdm
from fcmpy.expert_fcm.input_validator import type_check
from .nhl import HebbianLearning
from .update_weight import AhlWeightUpdate
from .update_state import FcmUpdateAsynch
from .termination import FirstCriterion
from .termination import SecondCriterion
from .update_parameters import Eta
from .update_parameters import Gamma


class AHL(HebbianLearning):
    """
        AHL algorithm for FCMs.
    """
    @type_check
    def __init__(self,  state_vector: dict, weight_matrix: pd.DataFrame,
                    activation_pattern:dict,  doc_values:dict) -> None:
        """
            Active Hebbian Learning for FCMs.

            Parameters
            ----------
            state_vector: dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept

            weight_matrix: pd.DataFrame
                                N*N weight matrix of the FCM.

            activation_pattern: dict
                                    sequence of activation concepts.
                                    keys ---> sequence step, values --> activated concept.

            doc_values: dict
                            Desired Output Concepts (DOCs) values.
                            keys ---> output concepts, values ---> desired output range ([min, max]).
                            e.g.,
                            doc_values = {'C1':[0.68,0.74], 'C5':[0.74,0.8]}
        """
        self.__state_vector = state_vector
        self.__weight_matrix = weight_matrix
        self.__doc_values = doc_values
        self.__activation_pattern = activation_pattern
        self.__weightUpdate = AhlWeightUpdate()
        self.__stateUpdate = FcmUpdateAsynch()
        self.__termination1 = FirstCriterion()
        self.__termination2 = SecondCriterion()
        self.__update_eta = Eta()
        self.__update_gamma = Gamma()
    
    @type_check
    def run(self, decay=0.03, learning_rate=0.01, iterations:int= 100, transfer:str= 'sigmoid', 
                    thresh:float = 0.002, l:Union[float, int]=0.98, auto_learn=False, **kwargs) -> pd.DataFrame:
        
        """
            Run the AHL algorithm for FCMs

            Parameters
            ----------
            decay: float
                    default --> 0.03
            
            learning_rate: float
                            default --> 0.01
            
            iterations: int
                        default --> 100
            
            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
                        default --> "sigmoid"
            
            thresh: float
                        default --> 0.002
            
            l: Union[float, int]
                    default --> 0.98
            
            auto_learn: bool
                        default --> False

            Return
            ------
            y: pd.DataFrame
                the optimized weight matrix
        """
        w_prior = self.__weight_matrix.copy()
        s_prior = self.__state_vector.copy()
        
        for _ in tqdm(range(iterations)):
            for cycle in self.__activation_pattern:
                
                if auto_learn:
                    learning_rate = self.__update_eta.update(cycle=(cycle+1), b1=kwargs['b1'], l1=kwargs['lbd1']) 
                    decay = self.__update_gamma.update(cycle=(cycle+1), b2=kwargs['b2'], l2=kwargs['lbd2']) 

                # Update state
                s_new = self.__stateUpdate.update(source=self.__activation_pattern[cycle], state_vector=s_prior,
                                                    weight_matrix=w_prior, transfer=transfer, l = l)
                # Update weights
                w_new = self.__weightUpdate.update(source=self.__activation_pattern[cycle], state_vector=s_new, 
                                                    weight_matrix=w_prior, eta=learning_rate, gamma=decay)
                # check termination
                if self.__termination1.terminate(doc_values=self.__doc_values, state_vector_prior=s_prior, state_vector_current=s_new) and \
                                                    self.__termination2.terminate(doc_values=self.__doc_values, state_vector_prior=s_prior, 
                                                                        state_vector_current=s_new, thresh=thresh):
                
                    print(f'The AHL learning process converged at step {_} with the learning rate eta = {learning_rate} and decay = {decay}!')
                    return w_new                   
                else:
                    w_prior = w_new
                    s_prior = s_new
        if _ >= iterations-1:
            warnings.warn(f'The AHL did not converge with the learning rate learning rate: {learning_rate} and decay: {decay}! Consider a different set of parameters.')
            return w_new