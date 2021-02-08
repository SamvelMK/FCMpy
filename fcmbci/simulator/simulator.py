import pandas as pd
import numpy as np
from simulator.inference import Inference
import warnings
from data_processor.input_validator import type_check

class Simulator(Inference):

    """
    The class includes methods for runing sumulations on top of a defined FCM.

    Methods:
            __init__(self)
            simulate(self, initial_state, weight_mat, transfer, inference, thresh=0.001, iterations=50, **params)
            add_inference_methods(self, func)
            remove_inference_methods(self, func_name)
            add_transfer_func(self, func)
            remove_transfer_func(self, func_name)
    """

    def __init__(self):
        super().__init__()
    
    @type_check
    def __getStableConcepts(self, weight_mat: np.ndarray) -> np.ndarray:

        """
        Extract the positions of the stable concepts (concepts with in-degree == 0).

        Parameters
        ----------
        weight_mat: numpy.ndarray
                        N*N weight matrix of the FCM.
        
        Return
        ----------
        y: numpy.ndarray
                the positions of the stable concepts (concepts with in-degree == 0)

        """

        stables = []
        for i in range(len(weight_mat)):
            if np.all(weight_mat[i] == 0):
                stables.append(i)

        return stables
    
    @type_check
    def simulate(self, initial_state: dict, weight_mat: np.ndarray, transfer: str, inference: str, thresh:float=0.001, iterations:int=50, **params) -> pd.DataFrame:
        
        """
        Runs simulations over the passed FCM.
        
        Parameters
        ----------
        initial_state: dict
                        initial state vector of the concepts
                        keys ---> concepts, values ---> initial state of the associated concept

        weight_mat: numpy.ndarray
                        N*N weight matrix of the FCM.

        transfer: str
                    transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

        inference: str
                    inference method --> "kosko", "mKosko", "rescaled"

        thresh: float
                    threshold for the error

        iterations: int
                        number of iterations
                        
        params: additional parameters for the methods

        Return
        ----------
        y: pandas.DataFrame
                results of the simulation.
        """
        
        results = pd.DataFrame(initial_state, index=[0])
        state_vector = np.array(list(initial_state.values()))

        # get the stable concept values
        stableConceptPos = self.__getStableConcepts(weight_mat=weight_mat.T)
        satble_values = state_vector[stableConceptPos]
        __infer = self.inference_methods[inference]
        __transfer = self.transfer_funcs[transfer]
        
        step_count = 0
        residual = thresh
        
        
        while step_count <= iterations:
            if (residual >= thresh):
                
                state_vector = __transfer(x=__infer(initial_state=state_vector, weight_mat=weight_mat, **params), **params)
                
                # Reset the stable values
                state_vector[stableConceptPos] = satble_values

                # Append the results
                results.loc[len(results)] = state_vector

                # update the step_count
                step_count +=1
                
                # compute the residuals between the steps.
                residual = max(abs(results.loc[len(results)-1] - results.loc[len(results) - 2]))
                
                if step_count >= iterations:
                    warnings.warn("The values didn't converged. More iterations are required!")
                else:
                    pass
                
            else: # if the residual < threshold print the step and exit the loop.
                print(f'The values converged in the {step_count+1} state (e <= {thresh})')
                break
        
        return results