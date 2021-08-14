import numpy as np
from abc import ABC, abstractmethod


class Convergence(ABC):
    """
        Class of FCM convergence methods
    """
    @abstractmethod
    def check_convergence() -> bool:
        raise NotImplementedError('Check_convergence method is not defined!')
    

class AbsDifference(Convergence):
    """
        Convergence check based on absolute difference .
    """
    @staticmethod
    def check_convergence(**kwargs) -> bool:
        """
            Compute the residuals (abs difference) of the outputConcepts
            between the simulation steps.
            
            Parameters
            ----------
            results: pd.DataFrame
                        the dataframe with the simulation results. 
                        Each row represents the results of a simulation step

            outputConcepts: None, list
                            if only specific outputConcepts should be considered
                            then a list of these concepts (in a string) should be passed.

            threshold: float, int
                        the threshold of the residuals to break the loop in the simulations.
            
            Return
            -------
            y: bool
                True if the residuals are <= the threshold, False if otherwise.
        """
        outputConcepts = kwargs['output_concepts']
        results = kwargs['results']
        threshold = kwargs['threshold']

        if outputConcepts:
            _ = results[outputConcepts]
            residual = max(abs(_.loc[len(results)-1] - _.loc[len(results) - 2]))
        else:
            residual = max(abs(results.loc[len(results)-1] - results.loc[len(results) - 2]))

        if residual <= threshold:
            return True
        else:
            return False