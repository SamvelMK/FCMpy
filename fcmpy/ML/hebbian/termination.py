import numpy as np
from abc import ABC
from abc import abstractmethod

class Termination(ABC):
    """
        Interface for terminating the learning algorithms
    """
    @abstractmethod
    def terminate() -> bool:
        """
            Determine whether to stop the learning process.
        """
        raise NotImplementedError('Terminate method is not defined!')


class FirstCriterion(Termination):
    """
        Termination condition based on F1 NHL
    """
    @staticmethod
    def __sumOfSquared(doc_values:dict, state_vector: dict):

        l = []
        for i in doc_values.keys():
            t = sum(doc_values[i])/len(doc_values[i])
            l.append(state_vector[0] - t)
        res = np.sqrt(sum([i**2 for i in l]))
        return res

    def terminate(doc_values:dict, state_vector_prior: dict, state_vector_current: dict) -> bool:
        """
            The objective of the training process is to find the set of
            weights that minimize function F1.
        
            Parameters
            ----------
            doc_values: dict
                            DOC values where the keys are the DOCs and 
                            the values are lists with min max values of the DOCs.

            state_vector_prior: np.ndarray
                            state vector at time t-1
            
            state_vector_current: np.ndarray
                            state vector at time t

            Return
            -------
            y: bool
                True if the condition is satisfied, False if otherwise.
        """
        prior = FirstCriterion.__sumOfSquared(doc_values=doc_values, state_vector=state_vector_prior)
        current = FirstCriterion.__sumOfSquared(doc_values=doc_values, state_vector=state_vector_current)

        if current < prior:
            return True
        else:
            return False


class SecondCriterion(Termination):
    """
        Termination method based on F2
    """
    @staticmethod
    def __absDifference(doc_values:dict, state_vector_prior: dict, state_vector_current: dict):
        """
            Calculate the termination condition based on the absolute difference.
        
            Parameters
            ----------
            doc_values: dict
                            DOC values where the keys are the DOCs and 
                            the values are lists with min max values of the DOCs.

            state_vector_prior: np.ndarray
                            state vector at time t-1
            
            state_vector_current: np.ndarray
                            state vector at time t

            Return
            -------
            y: float
                maximum absolute difference between the prior and current state values
        """
        prior = dict((key, state_vector_prior[key]) for key in doc_values.keys())
        current = dict((key, state_vector_current[key]) for key in doc_values.keys())

        dif = {key: abs(current[key] - prior.get(key, 0)) for key in prior}.values()
        max_dif = max(list(dif))

        return max_dif

    def terminate(doc_values:dict, state_vector_prior: dict, 
                        state_vector_current: dict, thresh:float=0.002) -> bool:
        """
            Calculate the termination condition based on the absolute difference.
        
            Parameters
            ----------
            doc_values: dict
                            DOC values where the keys are the DOCs and 
                            the values are lists with min max values of the DOCs.

            state_vector_prior: np.ndarray
                            state vector at time t-1
            
            state_vector_current: np.ndarray
                            state vector at time t
            
            thresh: float
                        threshold of absolute difference between the
                        prior and current values of the states

            Return
            -------
            y: bool
                True if the condition is satisfied, False if otherwise.
        """
        max_dif = SecondCriterion.__absDifference(doc_values, state_vector_prior, state_vector_current)
        
        if max_dif < thresh:
            return True
        else:
            return False