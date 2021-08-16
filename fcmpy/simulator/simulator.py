import pandas as pd
import numpy as np
import warnings
from typing import Union
from abc import ABC, abstractmethod
from fcmpy.store.methodsStore import InferenceStore
from fcmpy.store.methodsStore import TransferStore
from fcmpy.store.methodsStore import ConvergenceStore
from fcmpy.expert_fcm.input_validator import type_check


class Simulator(ABC):
    """
        Class of methods for simulating FCMs.
    """
    @abstractmethod
    def simulate():
        raise NotImplementedError('Simulate method is not defined!')


class FcmSimulator(Simulator):
    """
        The class includes methods for running simulations on top of a defined FCM.

        Methods:
            simulate(initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], 
                            transfer: str, inference: str, thresh:float=0.001, iterations:int=50, 
                            output_concepts = None, convergence = 'absDiff', **kwargs)
    """
    @staticmethod
    @type_check
    def __getStableConcepts(weight_matrix: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
            Extract the positions of the stable concepts (concepts with in-degree == 0).

            Parameters
            ----------
            weight_matrix: numpy.ndarray
                            N*N weight matrix of the FCM.
            
            Return
            ----------
            y: numpy.ndarray
                    the positions of the stable concepts (concepts with in-degree == 0)
        """
        stables = []
        for i in range(len(weight_matrix)):
            if np.all(weight_matrix[i] == 0):
                stables.append(i)

        return stables
    
    @staticmethod
    @type_check
    def __simInputTransform(weight_matrix: Union[pd.DataFrame, np.ndarray], initial_state: dict) -> tuple:
        """
            Transform the weight matrix and initial state vector for the simulation.
        """
        if type(weight_matrix) != np.ndarray:
            # Align the initial_vector order for correct computations (vec . mat)
            initial_state = {k : initial_state[k] for k in weight_matrix.columns}
            weight_matrix=weight_matrix.to_numpy()
        else:
            warnings.warn("When passing an initial state with a weight matrix type \
                numpy.ndarray make sure that the order of the keys in the dictionary \
                with the initial states matches the order of the column of the numpy.ndarray!")
        
        return weight_matrix, initial_state

    @staticmethod
    @type_check
    def __outputConceptCheck(output_concepts: Union[None, list], initial_state: dict):
        """
            Check if output concepts are in the initial_state.keys()
        """
        if output_concepts:
            r = set(output_concepts) - set(initial_state.keys())
            if r:
                raise ValueError(f'The specified output concept {r} is not in the list.')
    
    @staticmethod
    @type_check
    def simulate(initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray], 
                        transfer: str, inference: str, thresh:float=0.001, iterations:int=50, 
                        output_concepts = None, convergence = 'absDiff', **kwargs) -> pd.DataFrame:
        """
            Runs simulations over the passed FCM.
            
            Parameters
            ----------
            initial_state: dict
                            initial state vector of the concepts
                            keys ---> concepts, values ---> initial state of the associated concept

            weight_matrix: pd.DataFrame, np.ndarray
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method --> "kosko", "mKosko", "rescaled"

            thresh: float
                        threshold for the error

            iterations: int
                            number of iterations

            output_concepts: bool, list
                                the output concepts for the convergence check
                                default --> None

            convergence: str,
                            convergence method
                            default --> 'absDiff': absolute difference between the simulation steps

            kwargs: additional parameters for the methods (e.g., lambda (l) parameter for the sigmoid function)

            Return
            ----------
            y: pandas.DataFrame
                    results of the simulation.
        """
        # Check the output concepts.
        FcmSimulator.__outputConceptCheck(output_concepts=output_concepts,
                                        initial_state=initial_state)

        # Transform the inputs if necessary.
        weight_matrix, initial_state = FcmSimulator.__simInputTransform(weight_matrix=weight_matrix,
                                                                        initial_state=initial_state)

        # create the empty dataframe for the results
        results = pd.DataFrame(initial_state, index=[0])
        state_vector = np.array(list(initial_state.values()))

        # get the stable concept values
        stableConceptPos = FcmSimulator.__getStableConcepts(weight_matrix=weight_matrix.T)
        satble_values = state_vector[stableConceptPos]

        # get the methods for the simulation.
        transfer = TransferStore.get(transfer)()
        inference = InferenceStore.get(inference)()
        conv = ConvergenceStore.get(convergence)()
        
        # initialize params
        convergenceStatus = False
        step_count = 0

        for _ in range(iterations):
            if not convergenceStatus:    
                infered = inference.infer(initial_state=state_vector, weight_matrix=weight_matrix, params=kwargs) # Inference
                state_vector = transfer.transfer(x=infered, params=kwargs) # Apply transfer func on the results
                
                # Reset the stable values
                state_vector[stableConceptPos] = satble_values

                # Append the results
                results.loc[len(results)] = state_vector

                # update the step_count
                step_count +=1
                
                # compute the residuals between the steps.
                convergenceStatus = conv.check_convergence(output_concepts=output_concepts, results=results, threshold = thresh, params=kwargs) 
            else:    
                print(f'The values converged in the {step_count+1} state (e <= {thresh})')
                break

            if step_count >= iterations:
                warnings.warn("The values didn't converge. More iterations are required!")

        return results