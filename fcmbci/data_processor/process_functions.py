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

# def consistency_check(data):
    
#     """
#     Checks whether the sign of the raitings the paris of the concepts
#     are consistent across all the experts.
    
#     Parameters
#     ----------
#     data : OrderedDict,
#     """   

#     # Obtain the pairs of concepts. 1) create a flat data file with all the expert inputs. 
#     #                               2) set the index of the data From, To.
#     flat_data = pd.concat([data[i] for i in data], sort = False)
#     flat_data = flat_data.set_index(['From', 'To'])
#     indexes = set(flat_data.index) # a set of all concept pairs.

#     # For each pair of concepts 1) select the expert inputs 
#     #                           2) check whether the max of the list == to the min of the list
#     inconsistencies = []
#     for pair in indexes:
#         f = []
#         for expert in data:
#             d = data[expert].set_index(['From', 'To'])
#             if pair in list(d.index): # In case if the concept is not present in one of the expert's map. (Note Multiindex).
#                 l = d.loc[pair].dropna()
#                 if len(l) != 0:
#                     f.append(float(l.values))
#         if len(f) > 0:
#             if min(f) != max(f):
#                 inconsistencies.append(pair)
#     # In case of inconsistencies in raiting between the concepts across the experts raise a value error.
#     if len(inconsistencies) > 0:
#         raise ValueError(f'{inconsistencies} pairs were raited inconsistently across the experts. Check the data!')

def consistency_check(data):
    
    """
    Checks whether the sign of the raitings the paris of the concepts
    are consistent across all the experts.
    
    Parameters
    ----------
    data : OrderedDict,
    """   

    # Obtain the pairs of concepts. 1) create a flat data file with all the expert inputs. 
    #                               2) set the index of the data From, To.
    flat_data = pd.concat([data[i] for i in data], sort = False)
    flat_data.columns = [x.lower() for x in flat_data.columns]
    flat_data = flat_data.set_index(['from', 'to'])
    indexes = set(flat_data.index) # a set of all concept pairs.
    
# #     # For each pair of concepts 1) select the expert inputs 
# #     #                           2) check whether the max of the list == to the min of the list

    inconsistencies = []
    for pair in indexes:
        f = []
        val = flat_data.loc[pair].values
        if len(set(val[~np.isnan(val)])) > 1:
            inconsistencies.append(pair)
    if len(inconsistencies) > 0:
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