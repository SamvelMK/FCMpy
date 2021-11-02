import skfuzzy as fuzz
from abc import ABC, abstractclassmethod  
from fcmpy.expert_fcm.input_validator import type_check


class FuzzyMembership(ABC):
    """
        Fuzzy membership functions.
    """
    @abstractclassmethod
    def membershipFunction() -> dict:
        raise NotImplementedError('membershipFunction method is not defined')


class TriangularMembership(FuzzyMembership):
    """
        Triangular Fuzzy Membership Functions. 
    """
    @staticmethod
    @type_check
    def membershipFunction(**kwargs) -> dict:
        """
            Generate triangular membership functions.

            Other Parameters
            ----------------
            **linguisticTerms: dict,
                                terms and the associated parameters 
                                'abc' --> e.g., {'+VL': [0.25, 0.5, 0.75]}
            
            **universe: numpy.ndarray,
                        the universe of discourse                 

            Return
            -------
            y: dict,
                Generated membership functions. The keys are the 
                linguistic terms and the values are 1d arrays.
        """
        linguisticTerms = kwargs['linguistic_terms'] 
        universe = kwargs['universe'] 
        mfs = {}

        for term in linguisticTerms.keys():
            mfs[term] = fuzz.trimf(x = universe, abc=linguisticTerms[term])
        
        return mfs


class GaussianMembership(FuzzyMembership):
    """
        Gaussian Fuzzy Membership Functions. 
    """
    @staticmethod
    @type_check
    def membershipFunction(**kwargs) -> dict:
        """
            Generate Gaussian membership functions.

            Other Parameters
            ----------------
            **linguisticTerms: dict,
                                terms and the associated parameters 
                                'mean' and 'sigma' -> e.g., {'+VL': [0.25, 0.1]}}
            **universe: numpy.ndarray,
                        universe of discourse
                            
            Return
            -------
            y: dict,
                Generated membership functions. The keys are the linguistic terms 
                and the values are 1d arrays.
        """
        universe = kwargs['universe']
        linguisticTerms = kwargs['linguistic_terms'] 
        mfs = {}

        for term in linguisticTerms.keys():
            mfs[term] = fuzz.gaussmf(x = universe, mean=linguisticTerms[term][0],
                                        sigma = linguisticTerms[term][1])
        
        return mfs


class TrapezoidalMembership(FuzzyMembership):
    """
        Trapezoidal membership function.
    """
    @staticmethod
    @type_check
    def membershipFunction(**kwargs) -> dict:
        """
            Generate Trapezoidal membership functions.

            Other Parameters
            ----------------
            **linguisticTerms: dict,
                                terms and the associated parameters 'abcd' --> e.g., {'+VL': [0, 0.25, 0.5, 0.75]}
            **universe: numpy.ndarray,
                        universe of discourse

            Return
            -------
            y: dict,
                Generated membership functions. The keys are the linguistic terms and the values are 1d arrays.
        """
        linguisticTerms = kwargs['linguistic_terms'] 
        universe = kwargs['universe'] 
        mfs = {}
        for term in linguisticTerms.keys():
            assert linguisticTerms[term][0] <= linguisticTerms[term][1] <= linguisticTerms[term][2] <= linguisticTerms[term][3]
            mfs[term] = fuzz.trapmf(x = universe, abcd=linguisticTerms[term])
        
        return mfs