import math
import numpy as np 
from fcmpy.ML.nhl_algorithm import NHL
from fcmpy.ML.ahl_algorithm import AHL

def simulator(hl_type,nConcept,learning_rate,decay,A0,W_init,doc,lbd,e=None,mode=None,l1=None,l2=None,b1=None,b2=None,maxsteps=100):
    '''
    Runs the simulation
    :param hl_type: non hebbain learning nhl or active hebbian learning ahl
    :param nConcept: number of concept
    :param learning_rate: learning rate
    :param decay: decay coefficient
    :param A0: input vector of concept values
    :param W_init: initial weight matrix
    :param doc: DOC
    :param lbd: lambda parameter for sigmoid function
    :param e: value of max change of doc values between the steps
    :param mode: for AHL -> are we updating the hyperparams or keeping them constant
    :param l1: parameter for updating learing hyperparams in AHL
    :param l2: parameter for updating learing hyperparams in AHL
    :param b1: parameter for updating learing hyperparams in AHL
    :param b2: parameter for updating learing hyperparams in AHL
    :param maxsteps: maximum n of steps until the convergance must be achieved
    :return: dictionary of parameters which resulted in algorithm convergence
    in the case of NHL
    values = {'A':hebbian.A,
                 'W':hebbian.W,
                 'learning_rate':hebbian.n,
                 'decay coef':hebbian.gamma,
                 'steps':hebbian.steps,
                 'lbd':hebbian.lbd}
    in the case of AHL
        values = {'A':hebbian.A,
             'W':hebbian.W,
             'learning_rate':hebbian.n,
             'decay coef':hebbian.gamma,
             'steps':hebbian.steps,
             'lbd':hebbian.lbd,
             'mode':hebbian.mode,
             'decayparams':[hebbian.l2,hebbian.b2],
             'lrparams':[hebbian.l1,hebbian.b1]}
     '''
    values = None
    if hl_type == 'nhl':
        
        hebbian = NHL(nConcept = nConcept, n = learning_rate, gamma=decay,lbd = lbd,e=e)
    else:

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
        # check if the edge exist, if it doesnt, skipp it

        # update weights

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

    