import pandas as pd
import collections
from typing import Union
from fcmpy.expert_fcm.input_validator import type_check


class Transform:
    """
        Collection of data transformation (static) methods.
    """
    @staticmethod
    @type_check
    def calculateProportions(data: pd.DataFrame, conceptPair: tuple, nExperts:int) -> dict:
        """
            Calculate the proportions of answers to each linguistic term.

            Parameters
            ----------
            data: pandas.DataFrame
                        Data frame that contains all the expert 
                        inputs (i.e., "flattened" OrderedDict)
            
            conceptPair: tuple,
                            concept pair for which the activation parameter
                            should be constructed

            Return
            -------
            y: dict,
                keys ---> linguistic terms, values ---> proportion
                of expert ratings.
        """
        activation_parameter = {}
        activation_parameter = (data.loc[conceptPair].sum()/nExperts).to_dict()

        return activation_parameter

    @staticmethod
    @type_check
    def flatData(data: Union[dict, collections.OrderedDict]) -> pd.DataFrame:
        """
            Create a flat data from an ordered dictionary.

            Parameters
            ----------
            data: dict,
                    keys ---> expertId, values ---> pandas.DataFrame
            
            Return
            -------
            y: pandas.DataFrame
                data with all the expert inputs in one dataframe.
        """
        # Create a flat data with all of the experts' inputs.
        flat_data = pd.concat([data[i] for i in data], sort = False)
        flat_data.columns = [i.lower() for i in flat_data.columns]
        flat_data = flat_data.set_index(['from', 'to'])
        flat_data = flat_data.sort_index(level=['from','to'])

        return flat_data
