import pandas as pd
import numpy as np
from simulator.inference import Inference
import warnings

class FcmSimulator(Inference):
    
    """ 
    Runs simulations over the passed FCM.
    """

    def __init__(self):
        super().__init__()

    def simulate(self, initial_state, weight_mat, transfer, inference, thresh, iterations, **params):
        """
        Runs simulations over the passed FCM.
        
        Parameters
        ----------
        initial_state: numpy.array
                        initial state vector of the concepts
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
        weight_mat = weight_mat.T
        
        __infer = self.inference_methods[inference]
        __transfer = self.transfer_funcs[transfer]
        
        step_count = 0
        residual = thresh
        
        
        while step_count <= iterations:
            if (residual >= thresh):
                
                state_vector = __transfer(x=__infer(initial_state=state_vector, weight_mat=weight_mat, **params), **params)
                
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