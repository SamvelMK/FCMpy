import math
import numpy as np 
from nhl_algorithm import NHL
from ahl_algorithm import AHL

def simulator(hl_type,nConcept,learning_rate,decay,A0,W_init,doc, option,lbd,e=None,mode=None,l1=None,l2=None,b1=None,b2=None,maxsteps=100):
    
    values = None
    if hl_type == 'nhl':
        
        hebbian = NHL(nConcept = nConcept, n = learning_rate, gamma=decay,lbd = lbd,e=e)
    elif hl_type == 'ahl':
        
        hebbian = AHL(nConcept = nConcept, n = learning_rate, gamma=decay,lbd = lbd,e=e,mode=mode,l1=l1,l2=l2,b1=b1,b2=b2)
    
    
    
    # add nodes 
    for i in range(nConcept):
        if i in doc.keys():
            hebbian.add_node(i, A0[i], doc = True, doc_values = doc[i])
        else:           
            hebbian.add_node(i,A0[i])

    # add edges
    for i in range(nConcept):
        for j in range(nConcept):
            hebbian.add_edge(i,j,W_init[i,j])             


    while (hebbian.steps < maxsteps) and (not hebbian.termination()):
        # 1 update termination condition and check if we can already terminate (as condition of while loop)

        # 2 make a step, so 1st we need a place where to put our new value of the step, 
        hebbian.next_step()
        
        # 3 after calculating new weights, calculate new activation functions 

        
        hebbian.update_node()
    #             check if the edge exist, if it doesnt, skipp it

                # update weights 
        if hl_type=='nhl':
            hebbian.update_edge(option =1)
        else:
            hebbian.update_edge()
        hebbian.update_termination()

        # checking for the requirements, and if all edges have the same sign/orientation ISSUE !!!
    if hl_type=='nhl':
#     print(hebbian.A[20])
        if np.all(hebbian.sgn(hebbian.W[0]) == hebbian.sgn(hebbian.W[-1])) and hebbian.steps < maxsteps: 

            print('success')

            values = {'A':hebbian.A,
                 'W':hebbian.W,
                 'learning_rate':hebbian.n,
                 'decay coef':hebbian.gamma,
                 'steps':hebbian.steps,
                 'lbd':hebbian.lbd}
    elif hebbian.steps < maxsteps: 
        print('success')
        values = {'A':hebbian.A,
             'W':hebbian.W,
             'learning_rate':hebbian.n,
             'decay coef':hebbian.gamma,
             'steps':hebbian.steps,
             'lbd':hebbian.lbd,
             'mode':hebbian.mode,
             'decayparams':[hebbian.l2,hebbian.b2],
             'lrparams':[hebbian.l1,hebbian.b1]}
        
       

        
        
    return values

    