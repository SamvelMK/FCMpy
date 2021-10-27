from abc import ABC
from abc import abstractmethod
from random import random
import pandas as pd
import numpy as np
import tqdm
import os
import contextlib
from fcmpy.ml.hebbian.update_state import FcmUpdate

class Validation(ABC):
    @abstractmethod
    def validate(**kwargs):
        raise NotImplementedError('validate method is not defined!')


class HebbianValidate(Validation):
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
            

class ISE(Validation):
    """
        In-Sample Error
    """
    def __init__(self):
        self.__sim = FcmUpdate()

    def validate(self, **kwargs):
        """
            In Sample Error

            Parameters
            ----------
            initial_state: dict
                            initial state of the concepts

            weight_matrix: pd.DataFrame
                            Candidate FCM solution (N*N weight matrix).

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"
            l: int
                parameter for the sigmoid function

            data: pd.DataFrame
                    longitudinal data
            
            Return
            ------
            y: float
                in sample error
        """
        init_state = kwargs['initial_state']
        weight_matrix = kwargs['weight_matrix']
        transfer = kwargs['transfer']
        inference = kwargs['inference']
        l = kwargs['l']
        data = kwargs['data']

        sim_data = [init_state]
        s_prior = init_state
        for _ in range(len(data)-1):
            res = self.__sim.update(state_vector=s_prior, weight_matrix=weight_matrix, transfer=transfer,
                                            inference=inference, l=l)
            s_prior = res
            sim_data.append(res)
        error = (1/((len(data)-1)*len(data.keys())))*(np.sum(np.sum(np.abs(pd.DataFrame(sim_data)-data))))
        return error


class OSE(Validation):
    """
        Out of Sample Error
    """
    def __init__(self):
        self.__sim = FcmUpdate

    def validate(self, **kwargs):
        """
            Out of Sample Error
        
            Parameters
            ----------
            weight_matrix: pd.DataFrame
                            Candidate FCM solution (N*N weight matrix).

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"
            l: int
                parameter for the sigmoid function

            data: pd.DataFrame
                    longitudinal data
            
            low: float
                    the lower bound of the concept values
            
            high: float
                    higher bound of the concept values
            
            k_validation: int
                            number of samples to generate for the validation
            
            Return
            ------
            y: tuple
                out of sample error, standard deviation
        """

        weight_matrix = kwargs['weight_matrix']
        transfer = kwargs['transfer']
        inference = kwargs['inference']
        l = kwargs['l']
        data = kwargs['data']
        low = kwargs['low']
        high = kwargs['high']
        concepts = data.keys()
        k_validation = kwargs['k_validation']
        error = []
        sim_data = []
        for i in range(k_validation):
            init_state = {k:np.random.uniform(low=low, high=high) for k in concepts}
            res = [init_state]
            # Simulate data points
            for t in range(len(data)-1):
                _ = self.__sim.update(state_vector=init_state, weight_matrix=weight_matrix, transfer=transfer,
                                                inference=inference, l=l)
                res.append(_)
            res = pd.DataFrame(res)
            if len(sim_data) > 1:
                error.append(np.abs(sim_data[1:]-res[1:]))
            
            sim_data = res
        std = np.std(error)

        avg_error=(1/(k_validation*(len(data)-1)*len(data.keys())))*np.sum(error)
        
        return avg_error, std
        