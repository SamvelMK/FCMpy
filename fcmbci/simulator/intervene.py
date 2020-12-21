import pandas as pd
import numpy as np
from inference import Inference
from simulator import FcmSimulator
import warnings

class Intervene(FcmSimulator):
    def __init__(self):
        super().__init__()
        self.interventions = {}
        self.test_results = {}
    
    def initialize(self, initial_state, weights_df, transfer, inference, thresh, iterations, **params):
        """
        initialize the FCM.

        Parameters
        ----------
        initial_state: numpy.array
                        initial state vector of the concepts
        weight_mat: panda.DataFrame
                        causal weights between concepts
        transfer: str
                    transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
        inference: str
                    inference method --> "kosko", "mKosko", "rescaled"
        thresh: float
                    threshold for the error
        iterations: int
                        number of iterations
        params: additional parameters for the methods
        """
        self.weights_df = weights_df
        self.__weight_mat = self.weights_df.to_numpy()
        self.__initial_state = initial_state
        self.__transfer = transfer
        self.__inference = inference
        self.__thresh = thresh
        self.__iterations = iterations
        self.__params = params
        self.baseline = self.simulate(initial_state = self.__initial_state, weight_mat = self.__weight_mat,
                                                   transfer = self.__transfer, inference = self.__inference, thresh = self.__thresh, 
                                                   iterations = self.__iterations, **params).iloc[-1]

    def add_intervention(self, name, causal_weights, effectiveness):
        """
        add an intervention node with the associated causal weights to the FCM.

        Parameters
        ----------
        name: str
                name of the intervention
        causal_weights: dict
                            keys ---> concepts the intervention impacts, value: the associated causal weight
        effectiveness: float
                        the degree to which the intervention was delivered
        """

        intervention = {}
        intervention['efectiveness'] = effectiveness
        
        # construct a weight matrix for a given intervention
        temp = self.weights_df.copy(deep=True)
        temp['antecident'] = temp.columns
        temp.set_index('antecident', inplace=True)
        temp['intervention'] = 0
        temp.loc[len(temp)] = 0
        temp.rename(index = {temp.index[-1] : 'intervention'}, inplace = True)
        
        # add the intervention impacts
        for key in causal_weights.keys():
            temp.loc['intervention', key] = causal_weights[key]
            
        # construct the new state vector for a given intervention
        temp_vector = self.baseline.copy(deep=True)
        temp_vector = temp_vector.append(pd.Series({'intervention': effectiveness})).to_dict()
        
        # add the causal weight_mat for the intervention
        intervention['weight_mat'] = temp
        intervention['state_vector'] = temp_vector
        self.interventions[name] = intervention    
    
    def remove_intervention(self, name):
        """
        remove intervention

        Parameters
        ----------
        name: str
                name of the intervention
        """

        del self.interventions[name]

    def test_interventions(self, name):
        """
        test intervention case.

        Parameters
        ----------
        name: str
                name of the intervention
        """

        w = self.interventions[name]['weight_mat'].to_numpy()
        s = self.interventions[name]['state_vector']
        
        res = self.simulate(initial_state=s, weight_mat=w, transfer=self.__transfer, 
                             inference=self.__inference, thresh=self.__thresh, 
                             iterations=self.__iterations, **self.__params)
        
        self.test_results[intervention_case] = res.loc[len(res)-1][:-1]