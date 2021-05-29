import math
import numpy as np 
from nhl_algorithm import NHL

def simulator(nConcept,learning_rate,decay,A0,W_init,doc, option,lbd,e=None):
    
    values = None
    
    nhl = NHL(nConcept = nConcept, n = learning_rate, gamma=decay,lbd = lbd,e=e)
    # add nodes 
    for i in range(nConcept):
        if i in doc.keys():
            nhl.add_node(i, A0[i], doc = True, doc_values = doc[i])
        else:           
            nhl.add_node(i,A0[i])

    # add edges
    for i in range(nConcept):
        for j in range(nConcept):
            nhl.add_edge(i,j,W_init[i,j])             


    while (nhl.steps < 100) and (not nhl.termination()):
        # 1 update termination condition and check if we can already terminate (as condition of while loop)

        # 2 make a step, so 1st we need a place where to put our new value of the step, 
        nhl.next_step()
        
        # 3 after calculating new weights, calculate new activation functions 

        
        nhl.update_node()
    #             check if the edge exist, if it doesnt, skipp it

                # update weights 
        nhl.update_edge(option =1)

        nhl.update_termination()

        # checking for the requirements, and if all edges have the same sign/orientation ISSUE !!!

#     print(nhl.A[20])
    if np.all(nhl.sgn(nhl.W[0]) == nhl.sgn(nhl.W[-1])) and nhl.steps < 1000: 

        print('success')
        values = {'A':nhl.A,
                 'W':nhl.W,
                 'learning_rate':nhl.n,
                 'decay coef':nhl.gamma,
                 'steps':nhl.steps,
                 'lbd':nhl.lbd}


        
        
    return values

    