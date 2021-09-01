import numpy as np
from abc import ABC
from abc import abstractmethod
from fcmpy.expert_fcm.input_validator import type_check

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
        Termination method based on F1 NHL
    """
    @staticmethod
    @type_check
    def __sumOfSquared(doc_values:dict, state_vector: dict) -> float:
        """
            Calculate the sum of the squared errors between the doc at step K and step k+1.
            
            Parameters
            ----------
            doc_values: dict
                        Desired output concepts (DOC)
                        keys ---> concepts, values ---> list of maximum and minimum values of the DOCs

            state_vector: dict
                            State vector of the concepts
                            keys ---> concepts, values ---> State of the associated concept

            Return
            -------
            y: float
                sum of the squared errors
        """
        l = []
        for i in doc_values.keys():
            t = sum(doc_values[i])/len(doc_values[i])
            l.append(state_vector[i] - t)
        res = np.sqrt(sum([i**2 for i in l]))
        return res

    @staticmethod
    @type_check
    def __checkRange(doc_values:dict, state_vector:dict) -> bool:
        """
            Check if the doc values are in the desired range.
        
            Parameters
            ----------
            doc_values: dict
                        Desired output concepts (DOC)
                        keys ---> concepts, values ---> list of maximum and minimum values of the DOCs

            state_vector: dict
                            State vector of the concepts
                            keys ---> concepts, values ---> State of the associated concept

            Return
            -------
            y: Bool
                True if the DOCs are in the desired range, False if otherwise.
        """
        vals = dict((key, state_vector[key]) for key in doc_values.keys())
        ps = []
        for i in vals.keys():
            if doc_values[i][0] <= vals[i] <= doc_values[i][1]:
                ps.append(True)
        if len(ps) == len(doc_values.keys()):
            return True

    @staticmethod
    @type_check
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

        if (current < prior) and FirstCriterion.__checkRange(doc_values=doc_values, state_vector=state_vector_current):
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
            Calculate the maximum absolute difference between the desired output concepts (DOCs) at step K
            and step K+1.
        
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

    @staticmethod
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
                True if the maximum difference between the steps is < the threshold,
                False if otherwise.
        """
        max_dif = SecondCriterion.__absDifference(doc_values, state_vector_prior, state_vector_current)
        
        if max_dif < thresh:
            return True
        else:
            return False