import pandas as pd
import itertools
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skfuzzy
import matplotlib.pyplot as plt
import matplotlib
import re
import networkx as nx
import functools

class FcmVisualize:
    """
    Visualize different components of an FCM.
    """
    
    def mf_view(self,
                title = 'Causal Strength',
                legend_anchor=(0.95, 0.6),
                figsize = (10,5)):
        
        '''Visualizes the membership function of the causal relationships between the concepts of the FCMs.
        
        Parameters
        ----------        
        
        title : str,
                default --> 'Causal Strength'
        
        figsize : tuple, 
                    default --> (10, 5)
        
        legend_anchor : tuple,
                        default --> (0.95, 0.6)
        '''
        
        fig, (ax0) = plt.subplots(nrows= 1, figsize= (10, 5))


        for i in self.terms:
            ax0.plot(self.universe, self.terms[i], linewidth=0.4, label=str(i))
            ax0.fill_between(self.universe, self.terms[i], alpha=0.3)

        
        ax0.set_title(title)
        ax0.legend(bbox_to_anchor=legend_anchor)
        
        
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.get_xaxis().tick_bottom()
        ax0.get_yaxis().tick_left()

        plt.tight_layout()

    
    def term_freq_hist(self, concept_1, concept_2,
                       width = 0.4,
                       col = '#72a2d9',
                       title = "Experts' raitings of causal strenght",
                       figsize = (10,5)):
        
        """ Visualizes the raitings of the terms across all the experts.
        The x axis represents the linguistic terms and the y axis represents the 
        frequency of occurence of the linguistic term.

        Parameters
        ----------        
        concept_1, concept_2 : str,
                
        width : float,
                default --> 0.4,
                    the size of the bins.
                    
        col : str,
                default --> '#72a2d9',
                Hex code or a color string.
        """

        data = self.expert_data.loc[concept_1][concept_2].value_counts()/len(self.data.keys())
        data = data.to_dict()
        
        fig, (ax0) = plt.subplots(nrows= 1, figsize= (10, 5))
        
        if data.values():
            ax0.bar(data.keys(), data.values(),  width= width, color = col)
            
            ax0.set_title(title)       

            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            ax0.get_xaxis().tick_bottom()
            ax0.get_yaxis().tick_left()

            plt.tight_layout()
        
        else:
            ax0.set_title(f'{concept_1} and {concept_2} are not related!')       
        
    
    def defuzz_view(self, 
                    concept_1, concept_2, 
                    figsize = (10,5),
                    legend_anchor = (0.95, 0.6)):
        """
        Visualize the activated output membership function of a pair of concepts with the defuzzification line.
        
        Parameters
        ----------        
        concept_1, concept_2 : str,
                                The pair of concepts to show the defuzzification graph.
                
        figsize : tuple,
                    default --> (10, 5),
                    the size of the figure. 
                    
        legen_anchor : tuple,
                        default --> (0.95, 0.6),
                        the position of the legend.
        """
        
        fig, ax0 = plt.subplots(figsize=(10, 5))
        
        defuzz = abs(fcmbci.causal_weights.loc[concept_1][concept_2])
        if defuzz > 0:
            y_activation = fuzz.interp_membership(self.universe, self.aggregated[f'{concept_1}{concept_2}'], defuzz)  # for plot
            out = np.zeros_like(self.universe) 

            for i in fcmbci.terms:
                ax0.plot(self.universe, self.terms[i], linewidth=0.4, label=str(i))
                ax0.fill_between(self.universe, self.terms[i], alpha=0.0)

            ax0.fill_between(self.universe, out, self.aggregated[f'{concept_1}{concept_2}'], facecolor='#ffa500', alpha=0.7)
            ax0.plot([defuzz, defuzz], [0, y_activation], 'k', linewidth=1.5, alpha=0.9)
            ax0.set_title(f'Aggregated membership and result (line) for {concept_1} and {concept_2}')

            ax0.legend(bbox_to_anchor=legend_anchor)

            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            ax0.get_xaxis().tick_bottom()
            ax0.get_yaxis().tick_left()

            plt.tight_layout()
        else:
            ax0.set_title(f'{concept_1} and {concept_2} are not related!')
            
    def system_view(self, outcome_node=None):
        
        """
        Visualize the FCM system.
        
        Parameters
        ----------        
        outcome_node : str,
                        default --> None,
                        The outcome of interest.
        """
        
        # For positive and negative edges
        def col(weights):
            if min(weights) < 0:
                return plt.cm.RdBu
            else:
                return plt.cm.Blues

        G = self.system
        plt.figure(figsize=(15, 10)) 
        pos=nx.circular_layout(G)
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

        M = G.number_of_edges()

        cmap = col(weights)

        node_sizes = [len(v)*1000 for v in G.nodes()]
        edge_colors = [i for i in weights]
        edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        
        # to visualize the outcome node. 
        if outcome_node:
            
            outcome_node = [outcome_node]
            node_list = list(G.nodes())
            node_list.remove(outcome_node[0])

            nodes = nx.draw_networkx_nodes(G, 
                                           pos,
                                           nodelist= node_list,
                                           node_size=node_sizes, 
                                           node_color="#cfcfe1")


            node_outcome = nx.draw_networkx_nodes(G, 
                                                   pos,
                                                   nodelist=outcome_node,
                                                   node_shape='o',
                                                   node_size=node_sizes, 
                                                   node_color="#8c198c")

            node_outcome = nx.draw_networkx_nodes(G, 
                                                   pos,
                                                   nodelist=outcome_node,
                                                   node_shape='*',
                                                   node_size=node_sizes, 
                                                   node_color="white")
        else:
            nodes = nx.draw_networkx_nodes(G, 
                                           pos,
                                           node_size=node_sizes, 
                                           node_color="#cfcfe1")
            

        nx.draw_networkx_labels(G, pos = pos)

        edges = nx.draw_networkx_edges(
                                        G,
                                        pos,
                                        node_size=node_sizes,
                                        arrowstyle="->",
                                        arrowsize=10,
                                        edge_color=edge_colors,
                                        edge_cmap=cmap,
                                        width=2,
                                    )

        # set alpha value for each edge
        for i in range(M):
            edges[i].set_alpha(edge_alphas[i])

        pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
        pc.set_array(edge_colors)
        plt.colorbar(pc)


        ax = plt.gca()
        ax.set_axis_off()
        plt.show()        