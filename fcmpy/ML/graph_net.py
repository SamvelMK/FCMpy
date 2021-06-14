import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw(W_init,A0,show=True):


    # W_init = np.asarray([[0,-0.4,-0.25,0,0.3],[0.36,0,0,0,0],[0.45,0,0,0,0],[-0.9,0,0,0,0],[0,0.6,0,0.3,0]])
    # A0 = np.asarray([[0.4,0 .707,0.607,0.72,0.3]])
    W_init = np.round(W_init, decimals = 2)
    A0 = np.round(A0, decimals = 2)
    G = nx.DiGraph()
    for i in range(W_init.shape[0]):
        for j in range(W_init.shape[1]):
            if W_init[i,j] == 0:
                # if edge doesnt exist
                continue
            else:
                if W_init[i,j] > 0:
                    val = '#008000'
                else:
                    val = '#ff0000'
                G.add_edge(i,j,weight = W_init[i,j],color = val)



    pos = nx.layout.circular_layout(G,)


    node_sizes = (A0**4)*300
    M = G.number_of_edges()


    # colors = [G[u][v]['color'] for u,v in G.edges()]
    colors = nx.get_edge_attributes(G,'color').values()
    weights = [G[u][v]['weight'] for u,v in G.edges()]
#     print(colors)
    plt.figure(figsize = (20,16))
    nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1', edge_color = colors, width = weights, arrowsize=20, arrowstyle='simple')
    if show:
        nx.draw_networkx_edge_labels(G,pos,connectionstyle='arc3, rad = 0.1', edge_labels=nx.get_edge_attributes(G,'weight'),label_pos = 0.8, font_size=25, clip_on = False)
    nx.draw_networkx_nodes(G,pos,connectionstyle='arc3, rad = 0.1', label = A0)


#     nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    
    ax = plt.gca()
    ax.set_axis_off()

    # table
    # TODO change this A1 dependent on A0
#     A1 = [list(A0[0])[node] for node in range(len(list(A0[0])))]
    print(A0)
    cell_text = [[f'node {i}', A0[i]] for i in range(len(A0))]
    # rows = [f'value of node {i}' for i in range(len(A1))]
    print(cell_text)
    plt.table(cellText = cell_text,loc = 'best',colWidths = [0.1,0.1])
    plt.show()

    # todo delete plt.show() change on saving image and then add the funtion to generate video using these images
