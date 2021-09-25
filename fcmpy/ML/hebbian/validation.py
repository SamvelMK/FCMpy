from abc import ABC
from abc import abstractmethod
from random import random
import tqdm
import os
import contextlib
import pandas as pd

class Validate(ABC):
    @abstractmethod
    def validate(**kwargs):
        raise NotImplementedError('validate method is not defined!')


class HebbianValidate(Validate):
    def __init__(self, FcmSimulator):
        self.__sim = FcmSimulator()
        self.results = {}

    def __gen_state_vector(self, keys:list):
        """
            Randomly generate state vectors
            
            Parameters
            ----------
            keys: list
                    concepts of the FCM.
            
            Return
            ------
            y: dict
                keys ---> concepts, values ---> pseudo random numbers [0, 1]
        """
        return {k:random() for k in keys}

    def validate(self, n_validations:int, doc_values:dict, concepts:list,
                    weight_matrix:pd.DataFrame, transfer:str='sigmoid',
                    inference:str='mKosko', thresh:float=0.001, iterations:int=50, l:int=1, convergence = 'absDiff', **kwargs):
        """
            Validate the weight matrix by running FCM simulations with random initial conditions.

            Parameters
            ----------
            n_validations: int
                            number of validation steps

            doc_values: dict
                            Desired Output Concepts (DOCs) values.
                            keys ---> output concepts, values ---> desired output range ([min, max]).
                            e.g.,
                            doc_values = {'C1':[0.68,0.74], 'C5':[0.74,0.8]}

            concepts: list
                        list of concepts

            weight_matrix: pd.DataFrame,
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"

            thresh: float
                        threshold for the error

            iterations: int
                            number of iterations

            convergence: str,
                            convergence method
                            default --> 'absDiff': absolute difference between the simulation steps
        """
        self.results = {c:[] for c in doc_values.keys()}

        for _ in tqdm.tqdm(range(n_validations)):
            init_state = self.__gen_state_vector(keys=concepts)

            # Redirect the print call in the simulation module
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                res = self.__sim.simulate(initial_state=init_state, weight_matrix=weight_matrix, transfer=transfer,
                                            inference=inference, thresh=thresh, iterations=iterations, l=l, convergence=convergence, kwargs=kwargs).iloc[-1].to_dict()  
        
            for i in doc_values.keys():
                if doc_values[i][0] <= res[i] <= doc_values[i][1]:
                    for d in doc_values.keys():
                        self.results[d].append(res[d])
                else:
                    raise ValueError(f'The concept values are out of the desired bounds for the following initial conditions {init_state}')
            