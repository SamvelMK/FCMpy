#################################################################################################
##                                  External Functions                                         ##                                                                      
#################################################################################################

import pandas as pd
import numpy as np
import itertools

def valence_check(linguistic_term):
        
        """ This function checks the valence (i.e., the sign) of the causal weight (in a linguistic Term).
        The function returns -1 if the linguistic term is negative or +1 if otherwise.
        
        Parameters
        ----------
        linguistic_term : str,
                            A string of the linguistic term --> '±H'
        
        Return
        ----------
        y : int,
                - 1 if the Linguistic Term is negative, +1 if its positive and 0 if otherwise.
        """
        if linguistic_term != 0:
            if '-' in linguistic_term:
                return -1
            else:
                return +1
        else:
            return 0

def consistency_check(data):
    """
    Checks whether the sign of the raitings the paris of the concepts
    are consistent across all the experts.
    
    Parameters
    ----------
    data : OrderedDict,
    """   
    flat_data = pd.concat([data[i] for i in data], sort = False) # create a df based on all expert inputs (flat data)
    flat_data.columns = [x.lower() for x in flat_data.columns] # Make all the column names lower case.
    flat_data.set_index(['from', 'to'], inplace=True)# set the columns from,to as indexes. 
    flat_data.sort_index(inplace=True) # Sort the indexes for higher performance.
    indexes = set(flat_data.index) # obtain a set of all concept pairs.

    inconsistencies = []
    for pair in indexes:
        f = []
        val = flat_data.loc[pair].values # for each pair of concepts select all the expert inputs.
        if len(set(val[~np.isnan(val)])) > 1: # check if the values (the sign of the raitings) are different across the experts.
            inconsistencies.append(pair)
    if len(inconsistencies) > 0: # If inconsistencies exist rais a ValueError.
        raise ValueError(f'{inconsistencies} pairs of concepts were raited inconsistently across the experts. Check the data!')

def check_column(data):
    """
    Checks whether the dataframe includes From ---> To column. It raises an error, if the columns are not found. 
    
    Parameters
    ----------
    data : OrderedDict,
    """
    columns = pd.concat([data[i] for i in data], sort = False).columns
    for expert in data.keys():
        if ('from' not in [x.lower() for x in data[expert].columns]) | ('to' not in [x.lower() for x in data[expert].columns]):
            raise ValueError('Columns From --> To were not found. Check the data!')


def label_gen(names):
    """
    Generates trunkated labels if the labels include mroe then 3 characters
    """

    text = []
    names = str(names) # in case if the concept's are integers.
    string = names.strip('\?!\t\n')
    if (len(string) > 3) & (len(string.split(' ')) > 1):
        text.append("".join(e[0] for e in string.split(' ')))
    elif len(string.split(' ')) == 1:
        text.append("".join(e[:3] for e in string.split(' ')))
    else:
        text.append(string)
    return text[0]


def correct_inconsistencies():
    """
    Correct inconsistencies in the sign of causality between the expert ratings by taking the sign of the majority ratings.
    """
    pass