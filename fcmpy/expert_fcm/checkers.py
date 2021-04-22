###########################################################################
##              Class of methods to check the input data                 ##
###########################################################################

import numpy as np
import pandas as pd
from datetime import date 
from tqdm import tqdm
import collections
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.expert_fcm.transform import Transform
from typing import Union

class ConsistencyCheck:
    
    @staticmethod
    def checkConsistency(data: Union[dict, collections.OrderedDict]):

        """
        Extract inconsistent ratings for the given linguistic terms in the supplied data.
        The method writes out an excel file with the inconsistencies and raises a ValueError if inconsistencies were identified.
        
        Parameters
        ----------
        data : dict, collections.OrderedDict,

        """
        
        current_date=date.today()

        flat_data = Transform.flatData(data=data)
        pairs = set(flat_data.index) # a set of all concept pairs.
        
        incon = {}
        for pair in tqdm(pairs):
            val = {}
            for expert in data.keys():
                dat = data[expert].copy(deep=True)
                dat.columns = [x.lower() for x in dat.columns]
                dat = dat.set_index(['from', 'to']).replace(r'', np.nan)
                dat['na'] = np.nan
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
            print(f'{list(res.index)} pairs of concepts were raited inconsistently across the experts. For more information check the inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx')

class ColumnsCheck:

    @staticmethod
    @type_check
    def checkColumns(data: Union[dict, collections.OrderedDict]):
    
        """
        Checks whether the dataframe includes From ---> To column. It raises an error, if the columns are not found. 
        
        Parameters
        ----------
        data : dict
        """
        for expert in data.keys():
            if ('from' not in [x.lower() for x in data[expert].columns]) | ('to' not in [x.lower() for x in data[expert].columns]):
                raise ValueError('Columns From --> To were not found. Check the data!')