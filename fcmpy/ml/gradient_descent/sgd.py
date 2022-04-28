import numpy as np
import random
import copy
from abc import ABC, abstractmethod
from tqdm import tqdm
from fcmpy.store.methodsStore import InferenceStore
from fcmpy.store.methodsStore import TransferStore
from fcmpy.store.methodsStore import SolverStore
from fcmpy.store.methodsStore import LossStore
from fcmpy.store.methodsStore import DWStore


class GradientDescent(ABC):
    """
        Interface for gradient descent algorithms.
    """
    @abstractmethod
    def run(**kwargs):
        raise NotImplementedError('run method is not defined.')


class SGD(GradientDescent):
    """
        Gradient Descent algorithm for training FCMs.
    """
    def __init__(self, initial_matrix, data, loss = 'mse'):
        self.weight_matrix = initial_matrix
        self.__shape = self.weight_matrix.shape
        self.data = data
        self.__T = len(data[0])
        self.__delta_w = DWStore.get(method=loss).calculate
        self.__loss = LossStore.get(method=loss).compute
        self.loss = []
    
    def __fcmSimulate(self, state_vector:np.array, weight_matrix:np.array, 
                        inference:str, transfer:str, time_steps:int, **kwargs):
        """
            Forward simulate data using FCM inference function.

            Parameters
            ----------
            state_vector: np.array
                            values of the concepts

            weight_matrix: np.array
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function --> "sigmoid", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"
            
            time_steps: int
                        number of time points to simulate
            
            Return
            ------
            y: np.array
                simulated data points
        """
        data = np.zeros((time_steps, len(state_vector))) # Empty matrix
        data[0] = state_vector
        infer = InferenceStore.get(method=inference).infer
        trans = TransferStore.get(method=transfer).transfer

        for i in range(time_steps-1):
            data[i+1] = trans(x=infer(initial_state = data[i], weight_matrix=weight_matrix), params=kwargs)
        return data

    def run(self, batch_size:int=38, epochs:int=1000, solver='regular', transfer:str='sigmoid', inference:str='kosko',
            learning_rate:float=0.001, l=1, b1:float=0.9, b2:float=0.999, e:float=10**-9, threshold_loss:float = 0.0001, 
            threshold_change:float=0.00001):
        """
            Run the SGD algorithm to train FCMs.
            
            Parameters
            ----------
            batch_size: int
                        the batch size for processing the data.
                        default -> 38

            epochs:int
                    number of learning cycles to run
                    default -> 1000

            solver: str:
                    type of solvers to use ---> "regular", "adam", "adamax"
                    default -> 'regular'

            transfer:str:
                    transfer function ---> "sigmoid", "tanh" (Note: The f(x) should be differentiable!)
                    default'sigmoid'

            inference:str
                    inference method ---> "kosko", "mKosko", "rescaled"
                    default -> "kosko"
                    
            learning_rate:float
                            learning rate for the sgd
                            default -> 0.001,
            
            l:int
                A parameter that determines the steepness of the sigmoid function at values around 0.
                default -> 1

            b1:float:
                hyperparameter for the adam solver
                default -> 0.9

            b2:float
                hyperparameter for the adam solver
                default -> 0.999

            e:float
                very small positive number to avoide non zero devision for the adam solver
                default -> 10**-9

            threshold_loss:float
                            threshold for the desired error level (i.e., loss)
                            default -> 0.0001

            threshold_change:float
                                threshold for changes in the parameters (used for checking convergence.)
                                default -> 0.00001
        """
        self.res = copy.deepcopy(self.weight_matrix)
        mats_average = 0
        solver = SolverStore.get(method=solver).update
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            random.shuffle(self.data)
            errors_batch = []
            for start in range(0, len(self.data), batch_size):
                errors_obs = 0
                mats = np.zeros(self.__shape)
                stop = start + batch_size
                batch = self.data[start:stop]

                for obs in batch:
                    simulated = self.__fcmSimulate(state_vector=obs[0], weight_matrix=self.res, inference=inference, transfer=transfer, time_steps=self.__T, l=l)
                    errors_obs += self.__loss(observed=obs, predicted=simulated, n=len(batch))
                    for t in range(self.__T-1):
                        dw = self.__delta_w(data=obs[t+1], predicted=simulated[t+1], state_vector=obs[t], 
                                                weight_matrix=self.res, transfer=transfer, inference=inference, l=l)
                        mats += solver(delta_w=dw, learning_rate=learning_rate, b1=b1, b2=b2, e=e, epoch=epoch)

                self.res = np.clip(self.res + (mats/(len(batch))), -1,1)
                mats_average = sum(np.abs(sum((mats)/len(batch))))
                errors_batch.append(errors_obs)
            self.loss.append(sum(errors_batch)/len(errors_batch))
            
            pbar.set_postfix({'loss': self.loss[-1]})
            
            if (mats_average <= threshold_change) or (self.loss[-1] <= threshold_loss):
                return f"converged at epoch:{epoch}, loss: {self.loss[-1]}"
        
        return f"The SGD did not converge!"
                