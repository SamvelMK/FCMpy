import numpy as np
import skfuzzy
from data_processor.input_validator import type_check

class FuzzyMembership:

    """
    The class includes methods for generating membership functions.

    Methods:
            __init__(self)
            __trimf(mf_x, weight, **params)
            add_membership_func(self, func)
            remove_membership_func(self, func_name)
    """

    def __init__(self):
        self.membership_func = {'trimf' : self.__trimf}

    @staticmethod
    @type_check
    def __trimf(universe: np.ndarray, linguistic_terms: list, noCausality: str) -> dict:
        
        """
        Generate triangular membership functions.

        Parameters
        ----------
        linguistic_terms: list,
                           List of linguistic terms used to express causality between concepts.
                           Note that the number of linguistic terms should be even. A narrow interval around 0 (for no causality option) is added automatically.
        
        noCausality: str,
                        the term used in the data to expresses no causality
                        default ---> 'No-Causality'

        Return
        ---------
        y: dict,
        Generated membership functions. The keys are the linguistic terms and the values are 1d arrays.
        """

        number = len(linguistic_terms)
        limits = [universe.min(), universe.max()]
        universe_range = (limits[1] - limits[0])/2
        widths = [universe_range / (((number/2) - 1) / 2.)] * int(number)
        
        
        # Create the centers of the mfs for each side of the x axis and then merge them together.
        centers_neg = list(np.linspace(-1, 0, number//2).round(2)) 
        centers_pos = list(np.linspace(0, 1, number//2).round(2))
        centers = centers_neg + centers_pos
        
        abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]
        
        abcs[number//2] = ([0, 0, centers_pos[1]]) # + Very low 
        abcs[((number//2) -1)] = [centers_neg[-2], 0, 0] # - Very Low

        # add a narrow interval for no causality.
        linguistic_terms.insert(len(linguistic_terms)//2, noCausality)
        abcs.insert(len(abcs)//2, np.array([-0.001, 0, 0.001]))
        
        # get the terms for positive and negative very lows for further correction
        nVl = linguistic_terms[len(linguistic_terms)//2 -1]
        pVl = linguistic_terms[len(linguistic_terms)//2 +1]

        terms = dict()
        
        # Repopulate
        for term, abc in zip(linguistic_terms, abcs):
            terms[term] = skfuzzy.trimf(universe, abc)
        
        # set the extra 1 produced by the extra 0 in the universe of discourse to 0.
        terms[nVl][len(terms[nVl])//2] = 0
        terms[pVl][(len(terms[nVl])//2)-1] = 0

        return terms
    
    @type_check
    def add_membership_func(self, func: dict):
        
        """
        Add a fuzzy membership function.

        Parameters
        ----------
        func: dict,
                key is the name of the function, value is the associated function.
        """

        self.membership_func.update(func)
    
    @type_check
    def remove_membership_func(self, func_name: str):
        
        """
        Remove a fuzzy membershipfunction.

        Parameters
        ----------
        func_name: str
                    name of the function to be removed.
        """

        if 'FuzzyMembership.__' not in str(self.membership_func[func_name]):
            del self.membership_func[func_name]
        else:
            raise ValueError('Cannot remove a base function!')
