import numpy as np
import pandas as pd
import collections
from datetime import date 
from tqdm import tqdm
from typing import Union
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.expert_fcm.transform import Transform


class ConsistencyCheck:
    """
        Class of methods for checking the consistency of the data.
    """
    @staticmethod
    def checkConsistency(data: Union[dict, collections.OrderedDict]):
        """
            Extract inconsistent ratings for the given linguistic terms in the
            supplied data. The method writes out an excel file with the
            inconsistencies and prints out a message if inconsistencies 
            were identified.
            
            Parameters
            ----------
            data : dict, collections.OrderedDict
        """
        current_date=date.today()
        flat_data = Transform.flatData(data=data)
        pairs = set(flat_data.index) # a set of all concept pairs.
        incon = {}

        for pair in tqdm(pairs):
            val = {}
            for expert in data.keys():
                dat = data[expert].copy(deep=True)
                dat.columns = [x.lower() for x in dat.columns] # set columns to lower case.
                dat = dat.set_index(['from', 'to']).replace(r'', np.nan) # replace the empty cells to np.nan
                dat[[i for i in dat if '-' in i]] = dat[[i for i in dat if '-' in i]] * -1
                v = dat.loc[pair].values[np.logical_not(np.isnan(dat.loc[pair].values))]
                if len(v) > 0:
                    val[expert] = int(v)
            if len(set(list(val.values()))) > 1:
                incon[pair] = val

        if incon:
            res = pd.DataFrame(incon).T
            res.index.set_names(['from', 'to'], inplace = True)
            res.to_excel(f'inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx', na_rep='NA')
            print(f'{list(res.index)} pairs of concepts were rated inconsistently across the experts. \
                 For more information check the inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx')


class ColumnsCheck:
    """
        Check the columns of the data.
    """
    @staticmethod
    @type_check
    def checkColumns(data: Union[dict, collections.OrderedDict]):
        """
            Checks whether the dataframe includes From --> To column. 
            It raises an error, if the columns are not found. 
            
            Parameters
            ----------
            data : dict
        """
        for expert in data.keys():
            if ('from' not in [x.lower() for x in data[expert].columns]) | \
                    ('to' not in [x.lower() for x in data[expert].columns]):
                raise ValueError('Columns From --> To were not found. Check the data!')