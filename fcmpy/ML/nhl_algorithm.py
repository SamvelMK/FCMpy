import numpy as np
import math
import copy


class NHL:

    def __init__(self, nConcept, e=None, n=None, gamma=None, h=None, lbd=None, beta=None):
        self.n = n if n is not None else 0.1  # learning rate
        self.gamma = gamma if gamma is not None else 0.98  # decay coef
        self.h = h if h is not None else 0  # tanh coef
        self.lbd = lbd if lbd is not None else 1 # steepness of continuous function
        self.termination1 = False
        self.termination2 = False
        self.termination3 = False
        self.W = np.zeros([1, nConcept, nConcept])  # initial W matrix
        self.A = np.zeros([1, nConcept])  # initial A matrix
        self.doc = []  # list of indexes which nodes are doc nodes
        self.doc_values = {}  # np array with min max value or dictionay with 2 values for each key
        self.e = e if e is not None else 0.005  # termination2 coefficient
        self.steps = 0
        self.nConcept = nConcept
        # or create a dictionary with parameters that is being checked each sept (or e.g. 100 steps) if the algorithm
        # doesnt finish update the dictionary

    def add_node(self, node, val, doc=False, doc_values=None, prt=False):
        '''

        :param node: node index
        :param val: value of the node
        :param doc: bool val if it is DOC
        :param doc_values: doc value as [t_min,t_max]
        :param prt: print the array of the nodes

        Adding the nodes to the map, we should add all the nodes before adding edges
        make sure you add not more nodes then you define when creating nhl
        e.g.
        map = nhl(nConcepts = ...,...)
        map.add(1,0.8,doc = [0.7,0.86])
        '''
        # check if it is the beginning of the simulation
        if self.steps != 0:
            raise Exception('cannot change the node values during the simulation')

        # checking if the values are the numbers
        try:
            float(node)
            float(val)
        except ValueError:
            raise Exception('node and val have to be numbers')

        # add the val to the initial activitation values
        self.A[0, node] = val

        if doc is True:
            # check for the if DOC values were added
            if doc_values is None or len(doc_values) != 2:
                raise Exception('you have to define DOC upper and ')

            # check for the if DOC values were added
            try:
                float(doc_values[0])
                float(doc_values[1])
            except ValueError:
                raise Exception('lower and upper bound have to number in the array')

            self.doc.append(node)
            self.doc_values[node] = doc_values

        # if prt is true, show the values of A
        if prt is True:
            print(self.A)

    def add_edge(self, source, target, value, prt=False):
        '''

        :param source: index of the source node
        :param target: index of target node
        :param value: value of the edge (including sign
        :param prt: print the weight array

        Adding the edge between 2 nodes, we can define whether:
        - increament of source will cause increment of target ( value >0)
        - increament of source will cause decrement of target ( value <0)
        e.g.
        map = nhl(nConcepts = ...,...)
        map.add_node(1,0.8,doc=True,doc_values = [0.7,0.86])
        map.add_node(2,0.5)
        map.add_edge(1,2,-.8)
        '''

        # checking if node exist
        try:
            self.A[0,source]
            self.A[0, target]
        except IndexError:
            raise Exception('one of the nodes doesnt exist')

        try:
            float(value)
        except ValueError:
            raise Exception('value is not a number')

        self.W[0, source, target] = value

        # show values of W
        if prt is True:
            print(self.W)

    def sigmoid(self, x):
        '''
        :param x: input value
        :param lbd: normalizing param
        :param h: normalizing param
        :return: sigmoid activation function output

        Sigmoid function used for updating of the node value after each step
        '''
        # TODO try inintial lbd as 0.5, recommended in another article
        try:
            sigm = 1 / (1 + math.exp(-x * self.lbd))
        except OverflowError:
            print('number to big' + str(x))
    
        return sigm
    

    def update_node(self, function=None):
        '''
        :param i: which concept, i.e. which column in weights

        updates node each step, we can indicate which function we want to use as an update function,
        most of HB algorithms are using sigmoid function, so it is defined as default. However, since some articles are
        proposing use of tanh, we also included this option here
        e.g.
        map.update_node(1)
        '''
        edge = 0
#
        if function is None or function == 'sigmoid':

                
#             vf = np.vectorize(self.sigmoid)
           
            self.A[-1] = 1/(1 + np.exp(-self.lbd*(self.A[-2] + self.W[-2]@self.A[-2])))
            
            
        elif (function == 'tangens') or (function == 'tanh') or (function == 'tan'):
            self.A[self.steps, i] = math.tanh(A)

    def update_edge(self, option=None):
        '''
        :param i: index of the target node
        :param j: index of the source node
        option: 1,2,3

        in the literature many different functions were proposed, the function 1 and 2 are basically the same, however
        they in the option 2, we function sgn is not used
        the option 3 is taken from another articles
        e.g.
        map.update_edge(1,2,option = 2)
        '''

        # ignoring 0 edges and i = j
#         if (self.W[0, row,col] == 0) or (row==col):
#             return

        if (option is None) or (option == 1):
        # according to the original one should A target,source, target -> col,row,col x,y,x Wyx , Wji, same, weights[y + x * A.length] == Wxy , weights[x + y * A.length] = Wyx
#             print(self.A[-2].shape)
            sign = np.vectorize(self.sign)  
            self.W[-1] = (self.gamma*self.W[-2]+self.n*self.A[-2]*np.ones((self.nConcept,self.nConcept))*((self.A[-2]*np.ones((self.nConcept,self.nConcept))).T-self.sgn(self.W[0])*self.W[-2]*(np.ones((self.nConcept,self.nConcept))*self.A[-2])))*np.abs(self.sgn(self.W[0]))
            
            
            
            
#             self.W[-1] = np.add(self.gamma * self.W[-2],np.subtract(np.multiply(self.n *self.A[-2],self.A[-2].T),np.multiply(self.n                                 *self.A[-2],np.multiply(self.W[-2],self.A[-2]))*np.abs(sign(self.W[0]))*sign(self.W[0])))*np.abs(sign(self.W[0]))
#             nhl.W[-1,row,col] = nhl.gamma * nhl.W[-2,row,col] + nhl.n * nhl.A[-2,row] * (nhl.A[-2,col] - nhl.sign(nhl.W[-2,row,col]) * nhl.A[-2,row] * nhl.W[-2,row,col])

        elif option == 3:
            # Prediction of stroke probability occurrence based on fuzzy cognitive maps weight update
            # here self.n is called alpha, not sure if its the same alpha, 0 <alpha<0.1
            
            self.W[-1] = sign(self.W[0])*(np.add(self.gamma * np.abs(self.W[-2]),np.subtract(np.multiply(self.n *self.A[-2],self.A[-2].T),np.multiply(self.n*self.A[-2],np.multiply(self.W[-2],self.A[-2]))*np.abs(sign(self.W[0]))*sign(self.W[0])))*np.abs(sign(self.W[0])))
        
        elif option == 4:
             self.W[-1,row,col] =  self.gamma* self.W[-2,row,col] + self.n * self.A[-2,col] * (self.A[-2,row] - self.sign(self.W[-2,row,col]* self.A[-2,col]))
        elif option == 5:
            self.W[-1,row,col] =  self.gamma* self.W[-2,row,col] + self.n * self.A[-2,row] * (self.A[-2,row] - self.sign(self.W[-2,row,col]) * self.A[-2,col] *self.W[-2,row,col])

            
    def termination(self):
        '''
        :return: if termination conditionas are satifying

        checking if simulation is completed
        '''
        # changing for or
        if self.termination1 and self.termination2:# and self.termination3:
            return True
        else:
            return False

    def update_termination(self):
        '''
        updating termination conditions each step
        the values of termination conditions are False as default
        after each step the function is checking for 2 termination conditions i.e.
        - if the cost function value is decreasing
        - if change between doc is smaller than defined threshold (default is 0.02)
        :param step: which is the current step
        '''
        # check in nodes are in bounds
        #         for node in self.doc:
        #             if self.A[self.steps,node] > self.doc_values[node][1] or self.A[self.steps,node] < self.doc_values[node][0]:
        #                 return

        if self.steps < 2:
            #             print('too soon to finish')
            return
        
        #         1st termination condition
        
        self.termination1 = self.f1()

        # checking for either one or second term
        # check if there was a change between the DOC values greater than e
        # term2    
        if np.all([abs(self.A[-1]-self.A[-2])[i] < self.e for i in self.doc]):
            self.termination2 = True
        
        #term3 
        # this is not the termination condition for the learning process, it is for the simulation to see if DOC is within the bounds
        # it does not have to be fulfill during the learning process!!!!
#         rules = 0
#         for node in self.doc:
#             if self.A[-1,node] < self.doc_values[node][1] and self.A[-1,node] > self.doc_values[node][0]:
#                 rules += 1
#         if rules == len(self.doc):
#             self.termination3 = True

    @staticmethod
    def sgn(X):
        '''
        function used to veryfiy if the sig of the weight is the same for output weights as it is for input weights
        '''
        W = copy.deepcopy(X)
        with np.nditer(W, op_flags=['readwrite']) as it:
            for x in it:
                if x > 0:
                    x[...] = 1
                elif x < 0:
                    x[...] = -1
                elif x == 0:
                    x[...] = 0

        return W

    def next_step(self):
        '''
        next step of the function
        increasing the dimension of np array for A and W
        '''

        # increase step value and dimensions of A and W
        self.steps += 1

        # add new axis to W and A and assign the value to 0

        self.W = np.resize(self.W, [self.steps + 1, self.W.shape[1], self.W.shape[2]])
        self.W[self.steps] = np.zeros([self.W.shape[1], self.W.shape[2]])

        self.A = np.resize(self.A, [self.steps + 1, self.A.shape[1]])
        self.A[self.steps] = np.zeros([1, self.A.shape[1]])

    @staticmethod
    def sign(x):
        if x < 0:
            return -1
        if x == 0:
            return 0
        return 1

    
    def f1(self):
        score = 0
        for doc in self.doc_values.keys():
            t = sum(self.doc_values[doc])/2
            if (math.sqrt((self.A[-1,doc]-t)**2)<math.sqrt((self.A[-2,doc]-t)**2)) and (math.sqrt((self.A[-2,doc]-t)**2)<math.sqrt((self.A[-3,doc]-t)**2)):
                score +=1
        return score == len(self.doc)