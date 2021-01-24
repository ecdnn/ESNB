#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
from abc import ABCMeta, abstractmethod
from math import *


class Individual(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def evaluate(self):
        pass
    
# Domination check
def dominate(p, q):
    result = False
    for i, j in zip(p.obj, q.obj):
        if i < j:  # at least less in one dimension
            result = True
        elif i > j:  # not greater in any dimension, return false immediately
            return False
    return result


def non_dominate_sorting(population):
    dominated_set = {}
    dominating_num = {}
    rank = {}
    for p in population:
        dominated_set[p] = []
        dominating_num[p] = 0

    sorted_pop = [[]]
    rank_init = 0
    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominate(p, q):
                dominated_set[p].append(q)
                dominating_num[q] += 1
            elif dominate(q, p):
                dominating_num[p] += 1
                dominated_set[q].append(p)
        # rank 0
        if dominating_num[p] == 0:
            rank[p] = rank_init # rank set to 0
            sorted_pop[0].append(p)

    while len(sorted_pop[rank_init]) > 0:
        current_front = []
        for ppp in sorted_pop[rank_init]:
            for qqq in dominated_set[ppp]:
                dominating_num[qqq] -= 1
                if dominating_num[qqq] == 0:
                    rank[qqq] = rank_init + 1
                    current_front.append(qqq)
        rank_init += 1

        sorted_pop.append(current_front)

    return sorted_pop

def evaluation(population):
    # Evaluation
    [ind.evaluate() for ind in population]
    return population

# one point crossover
def one_point_crossover(p, q):
    gene_length = len(p.dec)
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    k = np.random.randint(gene_length)
    child1[:k] = p.dec[:k]
    child1[k:] = q.dec[k:]

    child2[:k] = q.dec[:k]
    child2[k:] = p.dec[k:]

    return child1, child2

# Bit wise mutation
def bitwise_mutation(p, p_m):
    gene_length = len(p.dec)
    # p_mutation = p_m / gene_length
    p_mutation = p_m
    for i in range(gene_length):
        if np.random.random()<p_mutation:
            p.dec[i] = not p.dec[i]
    return p


# Variation (Crossover & Mutation)
def variation(population, p_crossover, p_mutation):
    offspring = copy.deepcopy(population)
    len_pop = int(np.ceil(len(population) / 2) * 2) 
    candidate_idx = np.random.permutation(len_pop)

    # Crossover
    for i in range(int(len_pop/2)):
        if np.random.random()<=p_crossover:
            individual1 = offspring[candidate_idx[i]]
            individual2 = offspring[candidate_idx[-i-1]]
            [child1, child2] = one_point_crossover(individual1, individual2)
            offspring[candidate_idx[i]].dec[:] = child1
            offspring[candidate_idx[-i-1]].dec[:] = child2

    # Mutation
    for i in range(len_pop):
        individual = offspring[i]
        offspring[i] = bitwise_mutation(individual, p_mutation)

    # Evaluate offspring
    offspring = evaluation(offspring)

    return offspring

# Crowding distance
def crowding_dist(population):
    pop_size = len(population)
    crowding_dis = np.zeros((pop_size, 1))

    obj_dim_size = len(population[0].obj)
    # crowding distance
    for m in range(obj_dim_size):
        obj_current = [x.obj[m] for x in population]
        sorted_idx = np.argsort(obj_current)  # ascending order
        obj_max = np.max(obj_current)
        obj_min = np.min(obj_current)

        # keep boundary point
        crowding_dis[sorted_idx[0]] = np.inf
        crowding_dis[sorted_idx[-1]] = np.inf
        for i in range(1, pop_size - 1):
            crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                      1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                             obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
    return crowding_dis


# Environmental Selection
def environmental_selection(population, n):
    pop_sorted = non_dominate_sorting(population)
    selected = []
    for front in pop_sorted:
        if len(selected) < n:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # select individuals according crowding distance here
                crowding_dst = crowding_dist(front)
                k = n - len(selected)
                dist_idx = np.argsort(crowding_dst, axis=0)[::-1]
                for i in dist_idx[:k]:
                    selected.extend([front[i[0]]])
                break
    return selected


def initialization(pop_size, args_list):
    indObj = args_list['individual']
    population = []
    for i in range(pop_size):
        ind = indObj(args_list=args_list, apply_pk=(i%2==0))
        population.append(ind)

    return population


def save_population(population, path):
    pop_list = []
    for ind in population:
        pop_list.append([ind.dec, ind.obj])
    
    with open(path, 'wb') as f:
        pickle.dump(pop_list, f)

def load_population(population, path):
    with open(path, 'rb') as f:
        pop_loaded = pickle.load(f)
            
    for ind, ind_loaded in zip(population, pop_loaded):
        ind.dec = ind_loaded[0]
        ind.obj = ind_loaded[1]
    return population
    