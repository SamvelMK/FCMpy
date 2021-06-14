import pandas as pd
import numpy as np 
from typing import Union
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.expert_fcm.transform import Transform
from abc import ABC, abstractclassmethod
import collections

class Entropy(ABC):
    
    """
    Entropy of the expert inputs.
    """
    
    @abstractclassmethod
    def calculateEntropy(data: pd.DataFrame, activationParamter):
        raise NotImplementedError('calculateEntropy method is not defined')
    
class InformationEntropy(Entropy):
    
    @staticmethod
    @type_check
    def calculateEntropy(**kwargs) -> pd.DataFrame:

        """
        Calculate the information entropy of the expert ratings.

        Other Parameters
        ----------------
        **data: collections.OrderedDict
                ordered dictionary with the expert inputs
        
        Return
        ---------
        y: pandas.DataFrame,
            entropy of the concept pairs in expert ratings
        """

        data = kwargs['data']

        nExperts = len(data)
        flat_data = Transform.flatData(data=data)
        prop = {}
        for concepts in set(flat_data.index):
            activation_parameter = Transform.calculateProportions(data=flat_data, conceptPair=concepts, nExperts=nExperts) 
            prop[concepts] = activation_parameter

        entropy_concept = {}
        for concept in prop.keys():
            p = prop[concept].values()
            res = -1*sum([i*np.log2(i) for i in p if i != 0])
            res = abs(res)
            entropy_concept[concept] = res
        
        # Prepare a formated dataframe
        entropy_concept = {k:[v] for k,v in entropy_concept.items()}
        entropy_concept = pd.DataFrame(entropy_concept).T
        entropy_concept.index.set_names(['From','To'], inplace=True)
        entropy_concept.columns = ['Entropy']
        entropy_concept = entropy_concept.sort_index(level=[0,1])

        return entropy_concept