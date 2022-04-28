from abc import ABC, abstractmethod
from tqdm import tqdm
import random
import numpy as np
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
    def __init__(self, initial_matrix, data, loss = 'mse'):
        self.weight_matrix = initial_matrix
        self.__shape = self.weight_matrix.shape
        self.data = data
        self.__T = len(data[0])
        self.__delta_w = DWStore.get(method=loss).calculate
        self.__loss = LossStore.get(method=loss).compute
        self.loss = []
    
    def __fcmSimulate(self, state_vector, weight_matrix, 
                        inference, transfer, time_steps, **kwargs):
        """
            Simulated data based on the given initial 
            conditions and a connection matrix.
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
                    simulated = self.__fcmSimulate(state_vector=obs[0], weight_matrix=self.weight_matrix, inference=inference, transfer=transfer, time_steps=self.__T, l=l)
                    errors_obs += self.__loss(observed=obs, predicted=simulated, n=len(batch))
                    for t in range(self.__T-1):
                        dw = self.__delta_w(data=obs[t+1], simulated=simulated[t+1], state_vector=obs[t], 
                                                weight_matrix=self.weight_matrix, transfer=transfer, inference=inference, l=l)
                        mats += solver(delta_w=dw, learning_rate=learning_rate, b1=b1, b2=b2, e=e, epoch=epoch)

                self.weight_matrix = np.clip(self.weight_matrix + (mats/(len(batch))), -1,1)
                mats_average = sum(np.abs(sum((mats)/len(batch))))
                errors_batch.append(errors_obs)
            self.loss.append(sum(errors_batch)/len(errors_batch))
            
            pbar.set_postfix({'loss': self.loss[-1]})
            
            if (mats_average <= threshold_change) or (self.loss[-1] <= threshold_loss):
                return f"converged at epoch:{epoch}, loss: {self.loss[-1]}"
        
        return f"The SGD did not converge!"
                