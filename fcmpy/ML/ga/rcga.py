#!/usr/bin/env python
# coding: utf-8
# updated!!!!
import numpy as np
import copy
import tqdm.auto as tq
import matplotlib.pylab as plt
import matplotlib
matplotlib.use("TkAgg")

class rcga:
    '''
    RCGA algrithm for creating FCM based on the sample valuee,
    nConcepts - number of concepts (nodes), concetps: initial concepts values,
    Pmutation: probability of mutation (default 0.5), Precombination: probability of crossover (0.9),
    population_size (default 100), max_generations: max nubmer of steps (def 100000),
    numberofsteps - number of simulation steps, should be the same as in the historical data,
    maxfitness - fitness value after which learning process can be stopped   
    '''

    def __init__(self, nConcepts, concepts, Pmutation=None, Precombination=None, population_size=None,
                 max_generations=None, historicaldata=None, fcm=None,
                 numberofsteps=None, tournamentP=None, tournamentK=None, lbd=None,maxfitness=None):

        # GENERAL PARAMS
        # types of mutations are randomly choosen according to authors of the article W.Stach et al. 2005
        self.mutation_methods = ['random', 'nonuniform', 'Muhlenbein']
        # types of selection are randomly choosen according to authors of the article W.Stach et al. 2005
        self.selection_methods = ['rulette', 'tournament']
        # proability of cell mutatiing
        self.prob_mutation = 0.5 if Pmutation is None else Pmutation
        self.prob_recombination = 0.9 if Precombination is None else Precombination
        self.tournamentP = 1 if tournamentP is None else tournamentP
        self.tournamentK = 5 if tournamentK is None else tournamentK  # or 10....
        self.lbd = 1 if lbd is None else lbd  # this is the operator of the sigmoid function, in a lot of papers it's set to 1 (elpiniki), Stach suggested 5

        # GENERATION PROPERTIES
        # size of the population, number of chromosomes in each population
        self.population_size = 100 if population_size is None else population_size
        if self.population_size % 2 != 0:
            raise ValueError('Population size must be an EVEN number')
        # nmax number of generations
        self.max_generations = 100000 # 300000 if max_generations is None else max_generations
        self.current_gen = 0
        self.generations = np.zeros((self.population_size, nConcepts, nConcepts - 1))
        self.nConcepts = nConcepts

        # HISTORICAL DATA
        # historical data obtained from fcm simulations or observations (in the format columns - concepts, rows - simulation steps)
        if historicaldata is None and fcm is None:
            raise ValueError('Cannot run the learning process without previous FCM architecture or historical data!!!')
        self.data = historicaldata
        # fcm which we are optimizing
        self.fcm = fcm

        # FITNESS FUNCTION
        self.generation_fitness = np.zeros((1, self.population_size))
        self.maxfitness = 0.999 if maxfitness is None else maxfitness
        self.concepts_for_testing = concepts
        # number of steps we have to run the simulation in order to calculate fintess function (in Stach paper - 1 step)
        self.numberofsteps = 2  # 5 if numberofsteps is None else numberofsteps # suggested 1
        # termination conditions
        self.termination = False

    def intitialize(self):
        # initialize 1st population
        self.generations = np.random.uniform(low=-1, high=1,
                                                size=(self.population_size, self.nConcepts, self.nConcepts - 1))

    #     def zero_diag_multip(self,a):
    #         # in order to keep the self connections (node j to j) == 0, the easiest way is to multiply the matrix of ones with 0 diagonal, element wise with each generation
    #         # in my opinion it should be done before the fitness function is calculated or, in case of the first generation, right after creation
    #         singl = np.ones((self.nConcepts,self.nConcepts))
    #         np.fill_diagonal(singl,0)
    #         multiplier = np.ones((self.population_size,self.nConcepts,self.nConcepts))
    #         multiplier[:] = singl
    #         return np.multiply(a,multiplier)
    # -------------------- FITNESS OF THE GENERATION --------------------------------------

    def simulateFCM(self, concepts, weights, nsteps):
        # we have to simulate fcm with current weights in order to calculate fitness function
        # concepts should be given as a np.array((1,nConcepts))

        # VERY IMPORTANT
        # weights as np.array((nConcepts,nConcepts-1)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # we skip the diagonal == 0
        # print(self.nConcepts)
        assert weights.shape == (self.nConcepts, self.nConcepts - 1), 'wrong encoding'
        out = np.zeros((nsteps, concepts.shape[-1]))
        out[0] = concepts
        for j in range(1, nsteps):
            newvalues = np.zeros((concepts.shape[0]))
            for i in range(concepts.shape[0]):
                idx = list(range(concepts.shape[0]))
                idx.remove(i)
                newvalues[i] = round(1 / (1 + np.exp(-(concepts[i] + concepts[idx] @ weights[i]))), 8)
            # unfortunately using this way we will change the values of the concepts in the same time step, that is why we need to operate on more variables
            # BROOOOOO
            out[j] = newvalues
            concepts = newvalues
        return out[-1]

    def calculate_fitness(self, weights):
        # calculate fitness for each of the chromosome
        # difference
        alpha = 1 / (self.numberofsteps - 1) * self.nConcepts  # concepts_for_testing.shape[-1]
        # we are countin L1
        # let's say we have both historical data and fcm, so we can simply simulate with new weights and calculate difference to obtain the fitness function

        error = alpha * np.sum(
            np.abs(np.subtract(self.data[-1], self.simulateFCM(self.concepts_for_testing, weights, self.numberofsteps))))
        return 1 / (100 * error + 1)

    # -------------------- CROSSOVER  --------------------------------------

    def crossover(self):
        crossover_pairs = self.generations
        a = list(np.random.choice([False, True], p=[1 - self.prob_recombination, self.prob_recombination],
                                  size=self.population_size).astype(int) * range(self.population_size))
        a = list(filter(lambda x: x != 0, a))
        # we are applying one point corssover and mixing 1st with 2nd, 3rd with 4th and so on...
        for i in range(0, len(a), 2):  # population size (defaul 100), every even idx
            # choosing if the crossover will happen
            # 1 take two crossover pairs
            chromA = crossover_pairs[i]
            chromB = crossover_pairs[i + 1]
            # 2 flatten them
            chromA = np.reshape(chromA, (self.nConcepts * (self.nConcepts - 1)))
            chromB = np.reshape(chromB, (self.nConcepts * (self.nConcepts - 1)))
            # 3 randomly choose the 'crossing point'
            point = np.random.choice(range(self.nConcepts * (self.nConcepts - 1)))
            # 4 swap the values
            chromA[point:] = chromB[point:]
            chromB[:point] = chromA[:point]
            # 5 reshape to (nconcepts,nconcepts)
            chromA = np.reshape(chromA, (self.nConcepts, self.nConcepts - 1))
            chromB = np.reshape(chromB, (self.nConcepts, self.nConcepts - 1))
        # after crossover, crossover_pairs are the latest generation

        self.generations = crossover_pairs

    # -------------------- MUTATION --------------------------------------
    def mutation(self):
        mut = np.random.choice(['random','nonuniform'])#,'muhlenbein'])

        if mut =='random':
            self.randommutation()
        elif mut =='nonuniform':
            self.numutation()
        # if mut =='muhlenbein':
        #     self.muhlenbeinmutation()

    def randommutation(self):
        # applying mutation
        # choosing x % indexes for mutation
        a = list(np.random.choice([False, True], p=[1 - self.prob_mutation, self.prob_mutation], size=self.population_size).astype(int) * range(self.population_size))
        a = list(filter(lambda x: x != 0, a))
        for i in a:
            # muation is happening with probability

            # random method
            j = np.random.choice(range(self.nConcepts), size=1)
            k = np.random.choice(range(self.nConcepts - 1), size=1)

            self.generations[i, j,k] = np.random.uniform(-1,1)
            # k, m = np.random.choice(range(self.nConcepts - 1), size=2)
            # j, l = np.random.choice(range(self.nConcepts), size=2)
            # k, m = np.random.choice(range(self.nConcepts - 1), size=2)
            # # randomly choose 2 indexes
            # value1 = self.generations[i, j, k]
            # value2 = self.generations[i, l, m]
            # # and swap the values
            # self.generations[i, j, k] = value2
            # self.generations[i, l, m] = value1
    def numutation(self):
        # choosing p % of chromosomes in the generation
        a = list(np.random.choice([False, True], p=[1 - self.prob_mutation, self.prob_mutation],
                                  size=self.population_size).astype(int) * range(self.population_size))
        a = list(filter(lambda x: x != 0, a))
        # randomly choose max 3 elements in the chromosome and change their vals
        d = round((self.max_generations-self.current_gen)/(self.max_generations/2))
        for i in a:
            # randomly choosing d% of the elements to mutate, it decreases with the n of generations

            for change in range(d):
                j = np.random.choice(range(self.nConcepts), size=1)
                k = np.random.choice(range(self.nConcepts - 1), size=1)
                self.generations[i, j, k] = np.random.uniform(-1, 1)

        # TODO - Other mutation methods

    # -------------------- SELECTION OF THE BEST CANDIDATES FOR THE NEXT GENERATION --------------------------------------

    def selection(self):
        # selecting the candidates from the last generation to the new generation
        # as paper suggestd we are randomly choosing the way to choose gene for crossover
        cross = np.random.choice(['rulette', 'tournament'])
        if cross == 'rulette':
            crossover_pairs = self.rulette()
        elif cross == 'tournament':
            crossover_pairs = self.tournament()

    def rulette(self):
        # choosing candidates for crossover with probability according to the fitness function of each chromosome
        selection = np.zeros((self.population_size, self.nConcepts, self.nConcepts - 1))
        # initial probability list
        p = self.generation_fitness[-2] / np.sum(self.generation_fitness[-2])
        for i in range(self.population_size):
            # choice with probability, choosing index of chromosome
            selection[i] = self.generations[np.random.choice(list(range(self.population_size)), p=list(
                p))]  # 'last' population is still an array of zeros

            # self matining is prohibited (according to some articles, such as Muhlenbein1993)
            # if a pair (0,1),(2,3) and so on... is equal, we choose again
            # if i % 2 == 1:
            #     while (np.all(selection[i] == selection[i - 1])):
            #         selection[i] = self.generations[np.random.choice(list(range(self.population_size)), p=list(
            #             p))]  # 'last' population is still an array of zeros
        # selected chromosomes pass to next generation
        self.generations = selection

    def tournament(self):
        # we choose randomly k chromosomes from the generation, then we would choose the best one with probability p, the 2nd best with p*(1-p), 3rd best wih p*((1-p)^2) and so on
        # if p == 1, we would always choose the 'fittest one' from the k candidates
        selection = np.zeros((self.population_size, self.nConcepts, self.nConcepts - 1))

        for j in range(self.population_size):
            # choose k random chromosomes or rather their indexes
            candidates = np.random.choice(list(range(self.population_size)), size=self.tournamentK)
            # choosing candidate
            if self.tournamentP == 1:
                # get fitness of each candidate
                chosen = (0, 0)  # index,fitness
                for index in candidates:
                    if self.generation_fitness[-2, index] > chosen[1]:
                        chosen = (index, self.generation_fitness[-2, index])
            # choosing crossovers to create new gen
            selection[j] = self.generations[chosen[0]]
            # self matining is prohibited (according to some articles, such as Muhlenbein1993)
            # if a pair (0,1),(2,3) and so on... is equal, we choose again
            # if j % 2 == 1:
            #     while (np.all(selection[j] == selection[j - 1])):
            #         candidates = np.random.choice(list(range(self.population_size)), size=self.tournamentK)
            #         chosen = (0, 0)  # index,fitness
            #         for index in candidates:
            #             if self.generation_fitness[-2, index] > chosen[1]:
            #                 chosen = (index, self.generation_fitness[-2, index])
            #         # choosing crossovers to create new gen
            #         selection[j] = self.generations[chosen[0]]
        self.generations = selection

    # -------------------- check termination --------------------------------------

    def check_termination(self):
        # checking for termination conditions
        # 1 if max n of generations was reached
        # 2 fitness fucntion is dope, less than threshold, then choosing the best gene of the generation
        if self.current_gen <2:
            return
        elif (self.current_gen >= self.max_generations) or (np.any(self.generation_fitness[-2] >= self.maxfitness)):
            self.termination = True

            # -------------------- expands dimensions --------------------------------------

    def expand_dims(self):
        # making space for one more generations
        # self.generations = np.append(self.generations,
        #                              np.zeros((1, self.population_size, self.nConcepts, self.nConcepts - 1)), axis=0)
        # since each new gen needs space for fitness...
        self.generation_fitness = np.append(self.generation_fitness, np.zeros((1, self.population_size)), axis=0)

        # -------------------- RUNNING THE OPTIMIZATION PROCESS  --------------------------------------

    def run(self):
        # run the optimization process
        # if we start from 1st step, randomly initialize first generation
        self.intitialize()
        self.current_gen += 1
        # calculate fitness for 1st gen
        # there was some issue so better deepcopy before calling f
        for i in range(self.population_size):
            chromosome = copy.deepcopy(self.calculate_fitness(self.generations[i]))
            self.generation_fitness[0, i] = chromosome

        # update termination condition
        self.check_termination()

        # ploting fitness
        # interactive mode
        plt.ion()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(list(range(self.current_gen)), np.max(self.generation_fitness[0]))
        fig.canvas.draw()
        plt.show(block=False)
        # plt.show()
        # if it is not true
        while not (self.termination):

            # NEW GENERATION
            self.current_gen += 1
            # print(self.current_gen)
            if self.current_gen % 100 == 0:
                print(f'We are at {self.current_gen}/{self.max_generations}')
                print(f'max fitness function so far is {np.max(self.generation_fitness[-2])}')
                line1.set_xdata(list(range(self.current_gen-2)))
                line1.set_ydata(np.max(self.generation_fitness[:-1],axis=1))
                # re-drawing the figure
                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()
                plt.pause(0.02)

                # to flush the GUI events
                # fig.canvas.flush_events()
                # print(f'sample weights {self.generations[-1,30]}')

            # 1. expanding dims for new generation
            self.expand_dims()

            # 2. crossover with probability pCross
            self.crossover()
            # 3. mutate with probability pMutate
            self.mutation()

            # 4. calculate fitness
            for i in range(self.population_size):
                chromosome = self.calculate_fitness(copy.deepcopy(self.generations[i]))
                self.generation_fitness[-2, i] = chromosome

            # 5. selection process - > new generation is being created
            self.selection()

            # 6. update termination condiation
            self.check_termination()

        # return the most fitted candidate of last generation
        return self.generations[np.where(self.generation_fitness[-1] == np.max(self.generation_fitness[-1]))]


# TODO
'''
1. corret selection !!!!!!!!!!!!!!!!!
1. Other mutation methods
2. encode differently

3. Do we allow to create all the weights or not ??
4. Try other norm, maybe L2...? or not sadfjsaonafslnsfal'nsfna

'''


def simulateFCM(concepts, weights, nsteps):
    # we have to simulate fcm with current weights in order to calculate fitness function
    # concepts should be given as a np.array((1,nConcepts))
    # weights as np.array((nConcepts,nConcepts-1)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # we skip the diagonal == 0
    # assert weights.shape == (5, 4), 'wrong encoding'
    out = np.zeros((nsteps, concepts.shape[-1]))
    out[0] = concepts
    for j in range(1, nsteps):
        newvalues = np.zeros((concepts.shape[0]))
        for i in range(concepts.shape[0]):
            idx = list(range(concepts.shape[0]))
            idx.remove(i)
            newvalues[i] = round(1 / (1 + np.exp(-(concepts[i] + concepts[idx] @ weights[i]))), 8)
        # unfortunately using this way we will change the values of the concepts in the same time step, that is why we need to operate on more variables
        # BROOOOOO
        out[j] = newvalues
        concepts = newvalues
    return out
    
if __name__ == "__main__":
    # test 1
    # tank case
    # A0 = np.asarray([0.4, 0.707, 0.607, 0.72, 0.3])
    # W_init = np.asarray([[0.36, 0.45, -0.9, 0], [-0.4, 0, 0, 0.6], [-0.25, 0, 0, 0], [0, 0, 0, 0.3], [0.3, 0, 0, 0]])
    # testc = 5
    # nofsteps = 2
    #
    # historicaldata = simulateFCM(A0,W_init,nofsteps)

    # test 2
    nofsteps = 2
    testc =7
    W_init = np.asarray([[0,0,0,0.3,0.2,0],[0,0,0,0,0,0],[0.2,0,0,0.15,0.25,0],
                          [0.3,0.4,0.3,0.35,0.3,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0.45,0.35,0.3,0.4,0.25,0.2]])
    A0 = np.asarray([0.47, 0.51, 0.13, 0, 1, 0.37, 0.1])



    historicaldata = simulateFCM(A0, W_init, nofsteps)

    GA = rcga(testc,A0,historicaldata=historicaldata,numberofsteps=nofsteps)
    W_final = GA.run()
    # writing results
    # print(np.reshape(W_final,(100,testc*(testc-1))).shape,GA.generation_fitness.shape)


    np.savetxt('finalweights50p2nd.txt',np.reshape(W_final,(100,testc*(testc-1))), delimiter=',')
    np.savetxt('fitnessall50p2nd.txt',GA.generation_fitness, delimiter=',')
