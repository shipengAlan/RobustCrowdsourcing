#! /usr/bin/env python
# coding: utf-8
import copy
import random
import math
import numpy as np
import time
import json


def binaryPackage():
    w = [1, 5, 8, 5, 6, 7, 6, 5, 8, 3]
    v = [9, 7, 5, 8, 6, 6, 5, 8, 5, 1]
    s = 30
    f = [[0 for i in range(s + 1)] for i in range(len(w) + 1)]
    for i in range(1, len(w) + 1):
        for j in range(s + 1):
            select = f[i - 1][j]
            noselect = f[i - 1][j - w[i - 1]] + \
                v[i - 1] if j >= w[i - 1] else 0
            f[i][j] = max(select, noselect)
    print f[len(w)][s]
    return f[len(w)][s]


def robustCrowd():
    p = [9, 7, 5, 8, 6, 6, 5, 8, 5, 1]
    s = 30
    K = 10
    T = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    dict_task_skill = {}
    for i in range(len(T)):
        if T[i] == 1:
            dict_task_skill[i] = 0
    f = [[[0] for j in range(s + 1)] for i in range(len(p) + 1)]
    for i in range(len(p) + 1):
        for j in range(s + 1):
            f[i][j].append(copy.deepcopy(dict_task_skill))
    A = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
         [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
         [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
         [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
         [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
         [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]]
    for i in range(1, len(p) + 1):
        for j in range(s + 1):
            noselect = f[i - 1][j]
            if j >= p[i - 1]:
                select = calRobust(f[i - 1][j - p[i - 1]], A[i - 1])
                print "select:", select
                print "noselect:", noselect
                f[i][j] = maxRobust(select, noselect)
                print "result:", f[i][j]
            else:
                print "a noselect:", noselect
                f[i][j] = f[i - 1][j]
    print f[len(p)][s]


def maxRobust(noselect, select):
    if noselect[0] > select[0]:
        return noselect
    elif noselect[0] == select[0]:
        sum_a = 0
        sum_b = 0
        for k, v in noselect[1].items():
            sum_a += v
        for k, v in select[1].items():
            sum_b += v
        if sum_a > sum_b:
            return noselect
        else:
            return select
    else:
        return select


def calRobust(info, skills):
    dict_have_skill = copy.deepcopy(info[1])
    i = 0
    for k, v in dict_have_skill.items():
        dict_have_skill[k] += skills[k]
        if i == 0 or dict_have_skill[k] < min:
            min = dict_have_skill[k]
            i = 1
    return [min, dict_have_skill]


class Metaheuristic(object):

    """docstring for Metaheuristic"""

    def __init__(self, N, A, K, s, T, p):
        self.TempReduction = 0.95
        self.instance = InstaceRobustCrowd(N, A, K, s, T, p)

    def _variance(self, fitness_solutions):
        sum1 = 0.0
        sum2 = 0.0
        for i in range(len(fitness_solutions)):
            sum1 += fitness_solutions[i]
        mean = sum1 / len(fitness_solutions)
        for i in range(len(fitness_solutions)):
            sum2 += (mean - fitness_solutions[i])**2
        var = sum2 / len(fitness_solutions)
        return var**(0.5)

    def SimulatedAnnealingByTime(self, timeLimit, numInitialSolutions):
        # Generate initial solutions and select the one with the best fitness.
        # This is also used to choose an initial temperature value.
        fitness_solutions = []
        currentSolution = None
        currentFitness = 0.0
        instance = self.instance
        # generate some initial solution and calculate temperature
        for i in range(1, numInitialSolutions + 1):
            solution = instance.InitialSolution(0.5)
            solutionFitness = instance.objectiveFunction(solution)
            if currentSolution is None or solutionFitness > currentFitness:
                currentSolution = solution
                currentFitness = solutionFitness
            fitness_solutions.append(solutionFitness)
        temperature = self._variance(fitness_solutions) if self._variance(fitness_solutions) != 0 else random.random()*2 + 0.8
        print "start solution:", currentSolution, currentFitness
        print "temperature:", temperature
        BestSolution = currentSolution
        BestFitness = currentFitness

        IterationTimeStart = time.clock()
        IterationTime = time.clock()
        # LevelLength = instance.calLevelLength()
        while temperature > 0 and IterationTime - IterationTimeStart < timeLimit:
            # for level in range(LevelLength):
            newSolution = instance.getOneNeighborByRandom(currentSolution)
            newFitness = instance.objectiveFunction(newSolution)
            fitnessDiff = newFitness - currentFitness
            if fitnessDiff > 0:
                if newFitness >= BestFitness:
                    BestSolution = newSolution
                    BestFitness = newFitness
                currentSolution = newSolution
                currentFitness = newFitness
            else:
                u = random.random()
                if u <= math.exp(-1 * abs(fitnessDiff) / temperature):
                    if newFitness >= BestFitness:
                        BestSolution = newSolution
                        BestFitness = newFitness
                    currentSolution = newSolution
                    currentFitness = newFitness
            temperature = self.TempReduction * temperature
            IterationTime = time.clock()
        return BestFitness, BestSolution

    def SimulatedAnnealing(self, timeLimit, numInitialSolutions):
        # Generate initial solutions and select the one with the best fitness.
        # This is also used to choose an initial temperature value.
        fitness_solutions = []
        currentSolution = None
        currentFitness = 0.0
        instance = self.instance
        # generate some initial solution and calculate temperature
        for i in range(1, numInitialSolutions + 1):
            solution = instance.InitialSolution(0.5)
            solutionFitness = instance.objectiveFunction(solution)
            if currentSolution is None or solutionFitness > currentFitness:
                currentSolution = solution
                currentFitness = solutionFitness
            fitness_solutions.append(solutionFitness)
        temperature = self._variance(fitness_solutions) if self._variance(fitness_solutions) != 0 else random.random()*2 + 0.8
        print "start solution:", currentSolution, currentFitness
        print "temperature:", temperature
        BestSolution = currentSolution
        BestFitness = currentFitness

        IterationTime = 0
        # LevelLength = instance.calLevelLength()
        while temperature > 0 and IterationTime < timeLimit:
            IterationTime += 1
            # for level in range(LevelLength):
            newSolution = instance.getOneNeighborByRandom(currentSolution)
            newFitness = instance.objectiveFunction(newSolution)
            fitnessDiff = newFitness - currentFitness
            if fitnessDiff > 0:
                if newFitness >= BestFitness:
                    BestSolution = newSolution
                    BestFitness = newFitness
                currentSolution = newSolution
                currentFitness = newFitness
            else:
                u = random.random()
                if u <= math.exp(-1 * abs(fitnessDiff) / temperature):
                    if newFitness >= BestFitness:
                        BestSolution = newSolution
                        BestFitness = newFitness
                    currentSolution = newSolution
                    currentFitness = newFitness
            temperature = self.TempReduction * temperature
        return BestFitness, BestSolution

    def improveFirst(self, solution):
        best_fitness = self.instance.objectiveFunction(solution)
        neighbor_solution = solution
        for i in range(len(solution)):
            verse = 1 if solution[i] == 0 else 0
            neighbor_solution = copy.deepcopy(solution)
            neighbor_solution[i] = verse
            if self.instance.isFeasible(neighbor_solution):
                if self.instance.objectiveFunction(neighbor_solution) > best_fitness:
                    return neighbor_solution
        return neighbor_solution

    def improveOpt(self, solution):
        best_fitness = self.instance.objectiveFunction(solution)
        best_solution = solution
        for i in range(len(solution)):
            verse = 1 if solution[i] == 0 else 0
            neighbor_solution = copy.deepcopy(solution)
            neighbor_solution[i] = verse
            if self.instance.isFeasible(neighbor_solution):
                if self.instance.objectiveFunction(neighbor_solution) > best_fitness:
                    best_solution = neighbor_solution
                    best_fitness = self.instance.objectiveFunction(neighbor_solution)
        return best_solution

    # Basic Local Search (First)
    def localSearchByImproveFirst(self, solution, timeLimit=10):
        IterationTime = 0
        # solution = self.instance.InitialSolution(0.5)
        solution = self.instance.repairSolution(solution)
        while IterationTime < timeLimit:
            IterationTime += 1
            solution = self.improveFirst(solution)
            solution = self.instance.repairSolution(solution)
        return self.instance.objectiveFunction(solution), solution

    # Basic Local Search (Opt)
    def localSearchByImproveOpt(self, solution, timeLimit=10):
        IterationTime = 0
        # solution = self.instance.InitialSolution(0.5)
        solution = self.instance.repairSolution(solution)
        while IterationTime < timeLimit:
            IterationTime += 1
            solution = self.improveOpt(solution)
            solution = self.instance.repairSolution(solution)
        return self.instance.objectiveFunction(solution), solution

        # Basic Local Search (First)
    def localSearchByImproveFirstByTime(self, solution, timeLimit=10):
        IterationTime = 0
        # solution = self.instance.InitialSolution(0.5)
        IterationTimeStart = time.clock()
        solution = self.instance.repairSolution(solution)
        IterationTime = time.clock()
        while IterationTime - IterationTimeStart < timeLimit:
            solution = self.improveFirst(solution)
            solution = self.instance.repairSolution(solution)
            IterationTime = time.clock()
        return self.instance.objectiveFunction(solution), solution

    # Basic Local Search (Opt)
    def localSearchByImproveOptByTime(self, solution, timeLimit=10):
        IterationTime = 0
        # solution = self.instance.InitialSolution(0.5)
        IterationTimeStart = time.clock()
        solution = self.instance.repairSolution(solution)
        IterationTime = time.clock()
        while IterationTime - IterationTimeStart < timeLimit:
            solution = self.improveOpt(solution)
            solution = self.instance.repairSolution(solution)
            IterationTime = time.clock()
        return self.instance.objectiveFunction(solution), solution

    # Evolutionary Computation Algorithms
    def EvolutionaryComputation(self, timeLimit, populationsize, mutationProbability, repair_enable=1, localsearch_enable=1):
        # initial population
        population = []
        evaluation = []
        for i in range(populationsize):
            solution = self.instance.InitialSolution(0.5)
            # repair
            if repair_enable == 1:
                solution = self.instance.repairSolution(solution)
            population.append(solution)
        # Local Search
        if localsearch_enable != 0:
            for i in range(populationsize):
                if localsearch_enable == 1:
                    population[i] = self.localSearchByImproveFirst(
                        population[i])[1]
                elif localsearch_enable == 2:
                    population[i] = self.localSearchByImproveOpt(
                        population[i])[1]
                evaluation.append(
                    self.instance.objectiveFunction(population[i]))
        # sort solution by evaluaiton
        for i in range(populationsize-1):
            for j in range(populationsize-1):
                if evaluation[j] < evaluation[j+1]:
                    temp_eva = evaluation[j]
                    evaluation[j] = evaluation[j+1]
                    evaluation[j+1] = temp_eva
                    # exchange solution
                    temp_item = population[j]
                    population[j] = population[j+1]
                    population[j+1] = temp_item
        # get best
        best_solution = population[0]
        best_fitness = evaluation[0]
        # while
        IterationTime = 0
        while IterationTime < timeLimit:
            IterationTime += 1
            crossPoint = random.randint(0, self.instance.N - 1)
            mut1stPoint = random.randint(0, self.instance.N - 1)
            mut2ndPoint = random.randint(0, self.instance.N - 1)
            offspring_population = []
            offspring_evaluation = []
            for i in range(populationsize/2):
                # get parent1, parent2
                parent1 = population[min(min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)), min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)))]
                parent2 = population[min(min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)), min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)))]
                # crossover
                descend1 = parent2[0:crossPoint]
                descend2 = parent1[0:crossPoint]
                descend1 += parent1[crossPoint:]
                descend2 += parent2[crossPoint:]
                # mutation
                if random.random < mutationProbability:
                    descend1[mut1stPoint] = random.randint(0, 1)
                    descend1[mut2ndPoint] = random.randint(0, 1)
                    descend2[mut1stPoint] = random.randint(0, 1)
                    descend2[mut2ndPoint] = random.randint(0, 1)
                # repair
                if repair_enable == 1:
                    descend1 = self.instance.repairSolution(descend1)
                    descend2 = self.instance.repairSolution(descend2)
                # Local search
                if localsearch_enable == 1:
                    descend1 = self.localSearchByImproveFirst(descend1)[1]
                    descend2 = self.localSearchByImproveFirst(descend2)[1]
                elif localsearch_enable == 2:
                    descend1 = self.localSearchByImproveOpt(descend1)[1]
                    descend2 = self.localSearchByImproveOpt(descend2)[1]
                offspring_population.append(descend1)
                offspring_evaluation.append(self.instance.objectiveFunction(descend1))
                offspring_population.append(descend2)
                offspring_evaluation.append(self.instance.objectiveFunction(descend2))
            # sort
            for i in range(populationsize-1):
                for j in range(populationsize-1):
                    if offspring_evaluation[j] < offspring_evaluation[j+1]:
                        temp_eva = offspring_evaluation[j]
                        offspring_evaluation[j] = offspring_evaluation[j+1]
                        offspring_evaluation[j+1] = temp_eva
                        # exchange solution
                        temp_item = offspring_population[j]
                        offspring_population[j] = offspring_population[j+1]
                        offspring_population[j+1] = temp_item
            # merge two sets
            i = 0
            j = 0
            index = 0
            new_population = []
            new_evaluation = []
            while index < populationsize:
                if evaluation[i] > offspring_evaluation[j]:
                    new_evaluation.append(evaluation[i])
                    new_population.append(population[i])
                    i += 1
                else:
                    new_evaluation.append(offspring_evaluation[j])
                    new_population.append(offspring_population[j])
                    j += 1
                index += 1
            # iteration
            population = new_population
            evaluation = new_evaluation
            # get best
            if evaluation[0] > best_fitness:
                best_solution = population[0]
                best_fitness = evaluation[0]
        return best_solution, best_fitness

    def EvolutionaryComputationByTime(self, timeLimit, populationsize, mutationProbability, repair_enable=1, localsearch_enable=1):
        # initial population
        population = []
        evaluation = []
        for i in range(populationsize):
            solution = self.instance.InitialSolution(0.5)
            # repair
            if repair_enable == 1:
                solution = self.instance.repairSolution(solution)
            population.append(solution)
        # Local Search
        if localsearch_enable != 0:
            for i in range(populationsize):
                if localsearch_enable == 1:
                    population[i] = self.localSearchByImproveFirst(
                        population[i])[1]
                elif localsearch_enable == 2:
                    population[i] = self.localSearchByImproveOpt(
                        population[i])[1]
                evaluation.append(
                    self.instance.objectiveFunction(population[i]))
        # sort solution by evaluaiton
        for i in range(populationsize-1):
            for j in range(populationsize-1):
                if evaluation[j] < evaluation[j+1]:
                    temp_eva = evaluation[j]
                    evaluation[j] = evaluation[j+1]
                    evaluation[j+1] = temp_eva
                    # exchange solution
                    temp_item = population[j]
                    population[j] = population[j+1]
                    population[j+1] = temp_item
        # get best
        best_solution = population[0]
        best_fitness = evaluation[0]
        IterationTimeStart = time.clock()
        IterationTime = time.clock()
        # while
        while IterationTime - IterationTimeStart < timeLimit:
            crossPoint = random.randint(0, self.instance.N - 1)
            mut1stPoint = random.randint(0, self.instance.N - 1)
            mut2ndPoint = random.randint(0, self.instance.N - 1)
            offspring_population = []
            offspring_evaluation = []
            for i in range(populationsize/2):
                # get parent1, parent2
                parent1 = population[min(min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)), min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)))]
                parent2 = population[min(min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)), min(random.randint(0, populationsize-1), random.randint(0, populationsize-1)))]
                # crossover
                descend1 = parent2[0:crossPoint]
                descend2 = parent1[0:crossPoint]
                descend1 += parent1[crossPoint:]
                descend2 += parent2[crossPoint:]
                # mutation
                if random.random < mutationProbability:
                    descend1[mut1stPoint] = random.randint(0, 1)
                    descend1[mut2ndPoint] = random.randint(0, 1)
                    descend2[mut1stPoint] = random.randint(0, 1)
                    descend2[mut2ndPoint] = random.randint(0, 1)
                # repair
                if repair_enable == 1:
                    descend1 = self.instance.repairSolution(descend1)
                    descend2 = self.instance.repairSolution(descend2)
                # Local search
                if localsearch_enable == 1:
                    descend1 = self.localSearchByImproveFirst(descend1)[1]
                    descend2 = self.localSearchByImproveFirst(descend2)[1]
                elif localsearch_enable == 2:
                    descend1 = self.localSearchByImproveOpt(descend1)[1]
                    descend2 = self.localSearchByImproveOpt(descend2)[1]
                offspring_population.append(descend1)
                offspring_evaluation.append(self.instance.objectiveFunction(descend1))
                offspring_population.append(descend2)
                offspring_evaluation.append(self.instance.objectiveFunction(descend2))
            # sort
            for i in range(populationsize-1):
                for j in range(populationsize-1):
                    if offspring_evaluation[j] < offspring_evaluation[j+1]:
                        temp_eva = offspring_evaluation[j]
                        offspring_evaluation[j] = offspring_evaluation[j+1]
                        offspring_evaluation[j+1] = temp_eva
                        # exchange solution
                        temp_item = offspring_population[j]
                        offspring_population[j] = offspring_population[j+1]
                        offspring_population[j+1] = temp_item
            # merge two sets
            i = 0
            j = 0
            index = 0
            new_population = []
            new_evaluation = []
            while index < populationsize:
                if evaluation[i] > offspring_evaluation[j]:
                    new_evaluation.append(evaluation[i])
                    new_population.append(population[i])
                    i += 1
                else:
                    new_evaluation.append(offspring_evaluation[j])
                    new_population.append(offspring_population[j])
                    j += 1
                index += 1
            # iteration
            population = new_population
            evaluation = new_evaluation
            # get best
            if evaluation[0] > best_fitness:
                best_solution = population[0]
                best_fitness = evaluation[0]
            IterationTime = time.clock()
        return best_solution, best_fitness

class InstaceRobustCrowd(object):

    """docstring for InstaceRobustCrowd"""

    def __init__(self, N, A, K, s, T, p):
        # number of all workers
        self.N = N
        # matrix of every worker's skills, 1 for have, 0 for not
        self.A = A
        # number of kinds of skills
        self.K = K
        # budget
        self.S = s
        # skills that task needs
        self.T = T
        # every worker's price
        self.p = p
        self.levelLengthFactor = 0.25

    def isFeasible(self, solution):
        sum_price = 0
        for i in range(self.N):
            if solution[i] == 1:
                sum_price += self.p[i]
        if sum_price <= self.S:
            return True
        else:
            return False

    def repairSolution(self, solution):
        res_solution = copy.deepcopy(solution)
        while not self.isFeasible(res_solution):
            i = random.randint(0, len(res_solution) - 1)
            if res_solution[i] == 1:
                res_solution[i] = 0
        return res_solution

    def robustValue(self, cur_t):
        min_best = float('inf')
        for i in range(self.K):
            if self.T[i] == 1:
                if cur_t[i] < min_best:
                    min_best = cur_t[i]
        return min_best

    def objectiveFunction(self, solution):
        if not self.isFeasible(solution):
            return 0
        cur_t = [0 for j in range(self.K)]
        for i in range(self.N):
            if solution[i] == 1:
                for k in range(self.K):
                    if self.T[k] == 1:
                        cur_t[k] += self.A[i][k]
        min_best = float('inf')
        for i in range(self.K):
            if self.T[i] == 1:
                if cur_t[i] < min_best:
                    min_best = cur_t[i]
        return min_best

    def InitialSolution(self, random_probability):
        solution = []
        for i in range(self.N):
            if random.random() < random_probability:
                solution.append(1)
            else:
                solution.append(0)
        return solution

    def getOneNeighborByExchange(self, solution):
        neighbor = copy.deepcopy(solution)
        a = random.randint(0, self.N - 1)
        b = a
        while a == b:
            b = random.randint(0, self.N - 1)
        t = neighbor[a]
        neighbor[a] = b
        neighbor[b] = t
        return neighbor

    def getOneNeighborByRandom(self, solution):
        neighbor = copy.deepcopy(solution)
        a = random.randint(0, self.N - 1)
        neighbor[a] = 1 if neighbor[a] == 0 else 0
        return neighbor

    def getAllNeighborByRandom(self, solution):
        neighbors = []
        for i in range(self.N):
            temp = copy.deepcopy(solution)
            temp[i] = 1 if temp[i] == 0 else 0
            neighbors.append(temp)
        return neighbors

    def getAllNeighbor(self, solution):
        pass

    def calLevelLength(self):
        self.levelLength = int(
            self.levelLengthFactor * (self.N * (self.N - 1)))
        return self.levelLength


class DSofRobustCrowd(object):
    # deep Search can return optimal solution,
    # but the computation complexity is exponential

    """docstring for DSofRobustCrowd"""

    def __init__(self, N, A, K, s, T, p):
        self.N = N
        self.A = A
        self.K = K
        self.S = s
        self.T = T
        self.p = p
        self.best = 0
        self.best_select = []

    def deepSearch(self, i, cur_v, cur_w):
        if i >= self.N:
            cur_t = [0 for j in range(self.K)]
            for w in cur_w:
                for k in range(self.K):
                    if self.T[k] == 1:
                        cur_t[k] += self.A[w][k]
            min_robust = float('inf')
            for k in range(self.K):
                if self.T[k] == 1:
                    if cur_t[k] < min_robust:
                        min_robust = cur_t[k]
            if min_robust > self.best:
                self.best = min_robust
                self.best_select = cur_w
            return
        else:
            if cur_v + self.p[i] <= self.S:
                temp_cur_w = copy.deepcopy(cur_w)
                temp_cur_w.append(i)
                self.deepSearch(i + 1, cur_v + self.p[i], temp_cur_w)
            temp_cur_w = copy.deepcopy(cur_w)
            self.deepSearch(i + 1, cur_v, temp_cur_w)

    def getBestSelect(self):
        res = []
        j = 0
        for i in range(self.N):
            if i == self.best_select[j]:
                res.append(1)
                j += 1
            else:
                res.append(0)
        return res

    def getBestFitness(self):
        return self.best


if __name__ == "__main__":
    # read worker from file
    for skill_n in [10, 20]:
        f = open("workerSet"+str(skill_n), 'r')
        p = []
        A = []
        N = 0
        K = skill_n
        while 1:
            line = f.readline()
            if not line:
                break
            p_i = line.strip("\n").split(",")[1]
            A_i = line.strip("\n").split(",")[0].split(" ")
            A_i = [int(i) for i in A_i]
            p.append(int(p_i))
            A.append(A_i)
            N += 1
        f.close()
        T = []
        S = []
        ff = open("TaskSet"+str(skill_n), 'r')
        TN = 0
        while 1:
            line = ff.readline()
            if not line:
                break
            p_t = line.strip("\n").split(",")[1]
            A_t = line.strip("\n").split(",")[0].split(" ")
            A_t = [int(i) for i in A_t]
            S.append(int(p_t))
            T.append(A_t)
            TN += 1

        file_ds = open("deepSearch.txt", 'w')
        file_ds_solution = open("deepSearch_solution.txt", 'w')
        for t in range(TN):
            ds = DSofRobustCrowd(N, A, K, S[t], T[t], p)
            ds_time_start = time.clock()
            ds.deepSearch(0, 0, [])
            ds_time_end = time.clock()
            print "deepSearch:", ds.best, ds.best_select
            print str(ds_time_end-ds_time_start)
            file_ds.write(str(ds.best)+" "+str(ds_time_end-ds_time_start)+"\n")
            file_ds_solution.write(json.dumps(ds.best_select)+"\n")
            m = Metaheuristic(N, A, K, S[t], T[t], p)
            repeated_count = 10
            for run_time in [0.01, 0.1, 1, 2, 3]:
                file_BILOpt = open(str(run_time)+"BILOpt.txt", 'a+')
                file_BILFirst = open(str(run_time)+"BILFirst.txt", 'a+')
                file_SA = open(str(run_time)+"SA.txt", 'a+')
                file_EC = open(str(run_time)+"EC.txt", 'a+')
                file_BILOpt_solution = open(str(run_time)+"BILOpt_solution.txt", 'a+')
                file_BILFirst_solution = open(str(run_time)+"BILFirst_solution.txt", 'a+')
                file_SA_solution = open(str(run_time)+"SA_solution.txt", 'a+')
                file_EC_solution = open(str(run_time)+"EC_solution.txt", 'a+')
                list_fitness_BILOpt = []
                list_solution_BILOpt = []
                list_fitness_BILFirst = []
                list_solution_BILFirst = []
                list_fitness_SA = []
                list_solution_SA = []
                list_fitness_EC = []
                list_solution_EC = []
                for r in range(repeated_count):
                    fitness_BILOpt, solution_BILOpt = m.localSearchByImproveOptByTime(m.instance.InitialSolution(0.5), run_time)
                    fitness_BILFirst, solution_BILFirst = m.localSearchByImproveFirstByTime(m.instance.InitialSolution(0.5), run_time)
                    print "localsearch"
                    fitness_SA, solution_SA = m.SimulatedAnnealingByTime(run_time, 20)
                    print "SA"
                    solution_EC, fitness_EC = m.EvolutionaryComputationByTime(run_time, 50, 0.2)
                    print "EC"
                    print r
                    list_fitness_BILOpt.append(fitness_BILOpt)
                    list_solution_BILOpt.append(solution_BILOpt)
                    list_fitness_BILFirst.append(fitness_BILFirst)
                    list_solution_BILFirst.append(solution_BILFirst)
                    list_fitness_SA.append(fitness_SA)
                    list_solution_SA.append(solution_SA)
                    list_fitness_EC.append(fitness_EC)
                    list_solution_EC.append(solution_EC)
                file_BILOpt.write(str(float(sum(list_fitness_BILOpt))/repeated_count)+"\n")
                file_BILFirst.write(str(float(sum(list_fitness_BILFirst))/repeated_count)+"\n")
                file_SA.write(str(float(sum(list_fitness_SA))/repeated_count)+"\n")
                file_EC.write(str(float(sum(list_fitness_EC))/repeated_count)+"\n")
                file_BILOpt_solution.write(json.dumps(list_solution_BILOpt)+"\n")
                file_BILFirst_solution.write(json.dumps(list_solution_BILFirst)+"\n")
                file_SA_solution.write(json.dumps(list_solution_SA)+"\n")
                file_EC_solution.write(json.dumps(list_solution_EC)+"\n")
                file_BILOpt.close()
                file_BILFirst.close()
                file_SA.close()
                file_EC.close()
                file_BILOpt_solution.close()
                file_BILFirst_solution.close()
                file_SA_solution.close()
                file_EC_solution.close()
        file_ds.close()
        file_ds_solution.close()






    # robustCrowd()
    """
    N = 10
    p = [9, 7, 5, 8, 6, 6, 5, 8, 5, 1]
    s = 30
    K = 10
    T = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    A = [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1],  # 0
         [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],  # 1
         [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],  # 2
         [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],  # 3
         [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],  # 4
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # 5
         [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 6
         [0, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 7
         [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],  # 8
         [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]]  # 9
    
    ds = DSofRobustCrowd(N, A, K, s, T, p)
    ds.deepSearch(0, 0, [])
    print ds.best
    print ds.best_select
    
    m = Metaheuristic(N, A, K, s, T, p)
    m.instance.InitialSolution(0.5)
    print m.localSearchByImproveOpt(m.instance.InitialSolution(0.5), 1000)
    print m.localSearchByImproveFirst(m.instance.InitialSolution(0.5), 1000)
    
    m = Metaheuristic(N, A, K, s, T, p)
    # SA
    BestFitness, BestSolution = m.SimulatedAnnealing(5, 20)
    print BestFitness, BestSolution

    # deep search
    dp = DSofRobustCrowd(N, A, K, s, T, p)
    dp.deepSearch(0, 0, [])
    print dp.getBestFitness(), dp.getBestSelect()
    
    # EC
    print m.EvolutionaryComputation(20, 50, 0.2)
    

    # d = DynamicProgramming(N, A, K, s, T, p)
    # print d.dp()
    """
