import pandas as pd
import numpy as np
from fcmbci.simulator.simulation_functions import *
import warnings

class FcmSimulator:
    
    """ Runs simulations over the passed FCM.
        
        Parameters
        ----------
        initial_state : dict,
                        A dictionary of Concepts as keys and their initial states. ---> {'C1': 0.5, 'C2' : 0.4}.
                        The states take only values in the range of [0,1] for the sigmoid transfer function and [-1,1] for the hperbolic tangent.

        weights : Data frame with the causal weights.

        iterations : int,
                        Number of itterations to run in case if the system doesn't converge.
        inference : str,
                    default --> 'mk' -> modified kosko; available options: 'k' -> Kosko, 'r' -> Rescale.
                    Method of inference.
                    
        transfer : str,
                    default --> 's' -> sigmoid; available options: 'h' -> hyperbolic tangent; 'b' -> bivalent; 't' trivalent. 
                    transfer function.
        l : int,
            A parameter that determines the steepness of the sigmoid and hyperbolic tangent function at values around 0. 
        
        thresh : float,
                    default -->  0.001,
                    a thershold for convergence of the values.
        """


    def __init__(self, initial_state=None, weights=None, iterations = 50, inference = 'mk', 
                            transfer = 's', l = 1, thresh = 0.001):
        self.scenarios = {}
        if (initial_state is not None) | (weights is not None):
            results = self.simulate(initial_state, weights, iterations, inference, 
                                        transfer, l, thresh)
        
            # Finding the first fixed point with the initial state vector.
            self.scenarios['initial_state'] = results
            
            self.initial_equilibrium = results.loc[len(results) - 1]
        
    def simulate(self, state, weights, iterations = 50, inference = 'mk', 
                 transfer = 's', l = 1, thresh = 0.001):
        
        """ Runs simulations over the passed FCM.
        
        Parameters
        ----------
        State : dict,
                        A dictionary of Concepts as keys and their states. ---> {'C1': 0.5, 'C2' : 0.4}.
                        The states take only values in the range of [0,1] for the sigmoid transfer function and [-1,1] for the hperbolic tangent.

        weights : Data frame with the causal weights.

        iterations : int,
                        Number of itterations to run in case if the system doesn't converge.
        inference : str,
                    default --> 'mk' -> modified kosko; available options: 'k' -> Kosko, 'r' -> Rescale.
                    Method of inference.
                    
        transfer : str,
                    default --> 's' -> sigmoid; available options: 'h' -> hyperbolic tangent; 'b' -> bivalent; 't' trivalent. 
                    transfer function.
        l : int,
            A parameter that determines the steepness of the sigmoid and hyperbolic tangent function at values around 0. 
        
        thresh : float,
                    default -->  0.001,
                    a thershold for convergence of the values.

        Return
        ----------
        y : dataframe,
            dataframe with the results of the simulation steps.
        """

        results = pd.DataFrame(state, index=[0])

        step_count = 0
        residual = thresh
        state_vector = list(state.values())
        
        while step_count <= iterations:
            if (residual >= thresh):
                # Inference
                if inference == 'mk':
                    res = weights.mul(state_vector, axis=0).sum()+state_vector
                elif inference == 'k':
                    res = weights.mul(state_vector, axis=0).sum()
                elif inference == 'r':
                    res = weights.mul(([2*i-1 for i in state_vector]), axis=0).sum()+([2*i-1 for i in state_vector])                
                else:
                    raise ValueError('Unrecognized inference method!')
                # Apply the transfer function    
                if transfer == 's':
                    state_vector = [sig(i, l) for i in res]
                elif transfer == 'h':
                    state_vector = [np.tanh(i) for i in res]
                elif transfer == 'b':
                    state_vector = [bi(i) for i in res]
                elif transfer == 't':
                    state_vector = [tri(i) for i in res]
                else:
                    raise ValueError('Unrecognized transfer function!')

                # Append the results
                results.loc[len(results)] = state_vector
                # update the step_count
                step_count +=1
                # compute the residuals between the steps.
                residual = max(abs(results.loc[len(results)-1] - results.loc[len(results) - 2]))
                
                if step_count >= iterations:
                        warnings.warn("The values didn't converged. More iterations are required!")
            else: # if the residual < threshold print the step and exit the loop.
                print(f'The values converged in the {step_count+1} state (e <= {thresh})')
                break
        return results

    def test_scenario(self, scenario_name, state_vector, weights, iterations = 50, 
                        inference = 'mk', transfer = 's', l = 1, thresh = 0.001):

        sv = self.initial_equilibrium.to_dict()
        sv.update(state_vector) # updated state vector
        results = self.simulate(sv, weights, iterations, inference,
                                 transfer, l, thresh)

        self.scenarios[scenario_name] = results