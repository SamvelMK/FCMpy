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
    
    Parameters
    ---------- 
    fcmdata : fcmdata object,
              An istance of FcmDataProcessor class.  
    """

    def __init__(self, fcmdata):
        self.fcmdata = fcmdata
    
    def mf_view(self,
                universe = None,
                terms = None,
                title = 'Causal Strength',
                figsize = (10,5),
                legend_anchor=(0.95, 0.6)):
        
        if self.fcmdata is not None:
            universe = self.fcmdata.universe
            terms = self.fcmdata.terms
        else:
            universe = universe
            terms = terms
        
        '''Visualizes the membership function of the causal relationships between the concepts of the FCMs.
        
        Parameters
        ---------- 
        universe : array,
                    default --> None,
                    universe of discourse.

        terms : dict,
                membership functions to be visualized.

        title : str,
                default --> 'Causal Strength'
        
        figsize : tuple, 
                    default --> (10, 5)
        
        legend_anchor : tuple,
                        default --> (0.95, 0.6)
        '''
        
        plt.figure(figsize= figsize)
        axes = plt.axes()
        for i in terms:
            axes.plot(universe, terms[i], linewidth=0.4, label=str(i))
            axes.fill_between(universe, terms[i], alpha=0.5)

        axes.set_title(title)
        axes.legend(bbox_to_anchor=legend_anchor)
        
        
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.get_xaxis().tick_bottom()
        axes.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()
    
    def term_freq_hist(self, concept_1, concept_2,
                       width = 0.4,
                       col = '#36568B',
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
        
        title : str,
                default --> "Experts' raitings of causal strenght"
        
        figsize : tuple, 
                    default --> (10, 5)

        """

        expert_data = self.fcmdata.expert_data

        
        data = expert_data[concept_1, concept_2]
        
        plt.figure(figsize= figsize)
        axes = plt.axes()
        
        if max(list(data.values())) != 0:
            axes.bar(data.keys(), data.values(),  width= width, color = col)
            
            axes.set_title(title)       

            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.get_xaxis().tick_bottom()
            axes.get_yaxis().tick_left()

            plt.tight_layout()
            plt.show()
            
        else:
            axes.set_title(f'{concept_1} and {concept_2} are not related!')       
            plt.show()

    
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
                    
        legend_anchor : tuple,
                        default --> (0.95, 0.6),
                        the position of the legend.
              
        """
        universe = self.fcmdata.universe
        terms = self.fcmdata.terms
        causal_weights = self.fcmdata.causal_weights
        aggregated = self.fcmdata.aggregated

        plt.figure(figsize=figsize)
        axes = plt.axes()
        
        defuzz = causal_weights.loc[concept_1][concept_2]
        if defuzz != 0:
            y_activation = fuzz.interp_membership(universe, aggregated[f'{concept_1} {concept_2}'], defuzz)  # for plot
            out = np.zeros_like(universe) 

            for i in terms:
                axes.plot(universe, terms[i], linewidth=0.3, label=str(i)) # plots all the mfs. 
                axes.fill_between(universe, terms[i], alpha=0.4)
                
            axes.fill_between(universe, out, 
                             aggregated[f'{concept_1} {concept_2}'], 
                             facecolor='#36568B', alpha=1)
            
            axes.plot([defuzz, defuzz], [0, y_activation], 'k', linewidth=1.5, alpha=1)
            axes.text(defuzz-0.05, y_activation+0.02, f'{round(defuzz, 2)}',
                     bbox=dict(facecolor='red', alpha=0.5))

            axes.set_title(f'Aggregated membership and result (line) for {concept_1} and {concept_2}')

            axes.legend(bbox_to_anchor=legend_anchor)
            
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.get_xaxis().tick_bottom()
            axes.get_yaxis().tick_left()

            plt.tight_layout()
            plt.show()
        else:
            axes.set_title(f'{concept_1} and {concept_2} are not related!')
            plt.show()
    @staticmethod        
    def system_view(nxData, concept_states = None, target=None):
        
        """
        Visualize the FCM system.
        
        Parameters
        ---------- 
        concept_states : dict,
                            default --> None,
                            dictionary of concepts as keys and their initial states as values.       
        target : list,
                    default --> None,
                    The outcome/s of interest.
        """
        G = nxData
        
        # For positive and negative edges
        def col(weights):
            if min(weights) < 0:
                norm = (-1, 1)
                color = plt.cm.RdBu
                return norm, color
            else:
                norm = (0, 1)
                color = plt.cm.Blues
                return norm, color
            
        def col_nodes(node_attr):
            if min(node_attr) < 0:
                norm = (-1, 1)
                color = plt.cm.coolwarm
                return norm, color
            else:
                norm = (0, 1)
                color = plt.cm.Reds
                return norm, color
                
        # if the concept_states are passed then set it as node atrb for vis.
        if concept_states is not None:
            nx.set_node_attributes(G, concept_states, 'concept_states')
            edges,node_weights = zip(*nx.get_node_attributes(G,'concept_states').items())
            norm_nodes, cmap_nodes = col_nodes(node_weights)
        else:
            cmap_nodes = None
            norm_nodes = (None, None)
            
        pos=nx.circular_layout(G)
        edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        norm_edges, cmap_edges = col(edge_weights)
        node_sizes = [len(v)*1000 for v in G.nodes()] # proportional to the len of the labels.
        
        # set node color
        if (concept_states is not None) and (target is not None):
            node_colors = [i for i in concept_states.values()]
        elif concept_states is not None:
            node_colors = [i for i in concept_states.values()]
        else:
            node_colors = '#A79BB9'
            
        edge_colors = [i for i in edge_weights]
        
        # to visualize the outcome node. 
        if target:
            target = target
            node_list = list(G.nodes())
            nodes = nx.draw_networkx_nodes(G, 
                                           pos,
                                           nodelist= node_list,
                                           node_size=node_sizes, 
                                           node_color= node_colors,
                                           cmap = cmap_nodes,
                                           vmin= norm_nodes[0],
                                           vmax=norm_nodes[1])

            node_outcome = nx.draw_networkx_nodes(G, 
                                                   pos,
                                                   nodelist=target,
                                                   node_shape='*',
                                                   node_size=node_sizes, 
                                                   node_color="white",
                                                   vmin= norm_nodes[0],
                                                   vmax=norm_nodes[1])
        else:
            nodes = nx.draw_networkx_nodes(G, 
                                           pos,
                                           node_size=node_sizes, 
                                           node_color=node_colors,
                                           cmap = cmap_nodes,
                                           vmin= norm_nodes[0],
                                           vmax=norm_nodes[1])
            

        nx.draw_networkx_labels(G, pos = pos)

        edges = nx.draw_networkx_edges(
                                        G,
                                        pos,
                                        node_size=node_sizes,
                                        arrowstyle="->",
                                        arrowsize=10,
                                        connectionstyle='arc3, rad=0.1',
                                        edge_color=edge_colors,
                                        edge_cmap=cmap_edges,
                                        width=2,
                                        vmin= norm_edges[0],
                                        vmax=norm_edges[1])

        ax = plt.gca()
        ax.set_axis_off()

    @staticmethod
    def simulation_view(simulation_results, scenario, nxData=None, network_view = True, target = None,
                       figsize = (10, 5), legend_anchor = (0.97, 0.6), title = None):
            
            """
            Visualize the simulation results.
            
            Parameters
            ---------- 
            simulation_results : dict,
                                    a dictioonary of the the simulation results where the keys are the names of the scenario
                                    and the values are the simulation results. 

            scenario : str,
                        name of the scenario/results to be visualized.

            network_view : bool,
                            default --> True.
            target : list,
                        default --> None,
                        The outcome/s of interest.

            figsize : tuple,
                            default --> (10, 5)
            legend_anchor : tuple,
                            default --> (0.97, 0.6)
            title : str,
                    default -->   
            """
        
            def sim_view():

                plt.plot(simulation_results[scenario])
                axes = plt.gca()

                if title is None:
                    axes.set_title(f'Simulation Results for {scenario}')
                else:
                    axes.set_title(title)
                axes.legend(simulation_results[scenario].columns, bbox_to_anchor=legend_anchor)

                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                axes.get_xaxis().tick_bottom()
                axes.get_yaxis().tick_left()
                plt.tight_layout()
        
            if network_view == False:
                sim_view()
                plt.show()
            elif (network_view == True) & (nxData is None):
                raise ValueError('Network data is missing. Network_view can be True only if Network Data is supplied.')
            else:
                concept_states = simulation_results[scenario].loc[len(simulation_results[scenario]) -1].to_dict()            
                plt.figure(figsize = figsize)
                plt.subplot(121)
                FcmVisualize.system_view(nxData, concept_states, target)

                plt.subplot(122)
                sim_view()
                plt.show() 
