import matplotlib.pyplot as plt

class FcmVisualize:
    """
    Visualize different components of an FCM.
    """
    def mf_view(self, figsize = (10, 5), title = 'Causal Strength', legend_anchor = (0.95, 0.6)):
        '''Visualizes the membership function of the causal relationships between the concepts of the FCMs.
        
        Parameters
        ----------        
        figsize : tuple, 
                default --> (10, 5)
        
        title : str, 
        
        legend_anchor : tuple,
                        default --> (0.95, 0.6)
        '''
        fig, (ax0) = plt.subplots(nrows= 1, figsize= figsize)

        for i in self.terms:
            ax0.plot(self.universe, self.terms[i], linewidth=0.4, label=str(i))
            ax0.fill_between(self.universe, self.terms[i], alpha=0.3)

        ax0.set_title(title)
        ax0.legend(bbox_to_anchor=legend_anchor)

        # Turn off top/right axes
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.get_xaxis().tick_bottom()
        ax0.get_yaxis().tick_left()

        plt.tight_layout()
    
    def term_freq_hist(self, concept_1, concept_2, col = '#72a2d9', title = "Experts' Raitings"):
        """
        Visualize the Expert's raitings of the causal paths between a given two concepts.
        
        Parameters
        ----------        
        concept_1 : str, 
        
        concept_2 : str, 
        
        width : tuple,
                default --> (0.95, 0.6)

        col : str,
                default --> '#72a2d9'
        
        title : str,
                default --> title

        """
        data = self.expert_data.loc[concept_1][concept_2].value_counts()/len(self.data.keys())
        data = data.to_dict()
        
        fig, (ax0) = plt.subplots(nrows= 1, figsize= (10,5))
        ax0.bar(data.keys(), data.values(),  width= 0.4, color = col)
        
        ax0.set_title(title)
        
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.get_xaxis().tick_bottom()
        ax0.get_yaxis().tick_left()

        plt.tight_layout()

        plt.tight_layout()
                