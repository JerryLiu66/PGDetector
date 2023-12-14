import random as rd
import os
import numpy as np

from GraphData import *
from Utils import *
from copy import deepcopy
import json



def init_population(pop_size, seed_idx, nodes_attr):
    pop = []
    for i in range(pop_size):
        C = Comm()
        C.init_comm(seed_idx, nodes_attr)
        pop.append(C)
    return pop


def selection(pop, pop_size, nodes_attr, n = 6):
    new_pop = []
    # elite reservation
    C_best = min(pop, key = lambda x: x.fitness(nodes_attr))
    new_pop.append(C_best)

    # tournament competition
    for i in range(pop_size - 1):
        tournament_list = rd.sample(pop, n)
        C_tournament = min(tournament_list, key = lambda x: x.fitness(nodes_attr))
        new_pop.append(C_tournament)

    return new_pop


def expand(idx, C, G, edges_attr):
    """
    calculate the probability of node idx joining community C
    """
    w_in_C = w_in_nC = w_out_C = w_out_nC = 0
    min_depth, has_neigh = np.inf, False

    # for input edges
    for k in G.incident(idx, mode = "in"):
        src_idx = G.es[k].source

        if src_idx in C.members:
            w_in_C += edges_attr[(src_idx, idx)]["weight"]
            min_depth = min(min_depth, C.depth[src_idx])
            has_neigh = True
        else:
            w_in_nC += edges_attr[(src_idx, idx)]["weight"]

    # for output edges
    for k in G.incident(idx, mode = "out"):
        trg_idx = G.es[k].target

        if trg_idx in C.members:
            w_out_C += edges_attr[(idx, trg_idx)]["weight"]
            min_depth = min(min_depth, C.depth[trg_idx])
            has_neigh = True
        else:
            w_out_nC += edges_attr[(idx, trg_idx)]["weight"]

    if not has_neigh:
        return 0, -1

    p_in = (w_in_C / (w_in_C + w_in_nC)) if (w_in_C + w_in_nC) != 0 else 0
    p_out = (w_out_C / (w_out_C + w_out_nC)) if (w_out_C + w_out_nC) != 0 else 0
    p = max(p_in, p_out)
    return p, min_depth


def crossover(C1, C2, G, nodes_attr, edges_attr):
    """
    crossover
    """
    C_cross = deepcopy(C1)
    difference_members = C2.members - C1.members

    for idx in difference_members:
        p, dp = expand(idx, C1, G, edges_attr)
        if p > rd.random(): C_cross.push_node(idx, dp + 1, G, nodes_attr, edges_attr)

    return C_cross


def mutation(C, G, important_nodes_idx, nodes_attr, edges_attr):
    """
    mutation
    """
    C_mutation = deepcopy(C)
    rand_num = rd.random()
    if len(C.members) == 1: rand_num = 1.0

    # add nodes
    if rand_num > 0.5:
        neigh_idx = set()
        for idx in C.members: neigh_idx |= set(G.neighbors(idx))
        neigh_idx -= C.members

        neigh_idx = neigh_idx & set(important_nodes_idx)
        for idx in neigh_idx:
            if idx not in C_mutation.members:
                p, dp = expand(idx, C, G, edges_attr)
                if p > rd.random(): C_mutation.push_node(idx, dp + 1, G, nodes_attr, edges_attr)

    # delete nodes
    else:
        for idx in C.deepest_members():
            if idx != C.seed_idx:
                p, _ = expand(idx, C, G, edges_attr)
                if (1 - p) > rd.random(): C_mutation.pop_node(idx, G, nodes_attr, edges_attr)

    return C_mutation


def GA(pop_size, converge_num, seed, etherdata, save_path):
    """
    genetic algorithm
    """
    seed_idx = etherdata.add2idx[seed]
    important_nodes_idx = etherdata.important_nodes_idx

    G = etherdata.G
    add2idx = etherdata.add2idx
    idx2add = etherdata.idx2add
    nodes_attr = etherdata.nodes_attr
    edges_attr = etherdata.edges_attr

    # ==== population initation ====
    pop = init_population(pop_size, seed_idx, nodes_attr)
    comm_size_line, fitness_best_line, fitness_avg_line = [], [], []

    converge = 0
    iter = 0
    members_best = set()

    while converge < converge_num:
        iter += 1
        print("Iteration {}".format(iter))

        # ==== Step 1: selection ====
        new_pop = selection(pop, pop_size, nodes_attr)
        print("Finish Selection.")

        # ==== Step 2: crossover ====
        p_cross = 0.5
        if iter == 1: p_cross = 0.0
        n = len(new_pop)

        for i in range(n):
            if p_cross >= rd.random():
                j = rd.randint(0, n - 1)
                C_cross = crossover(new_pop[i], new_pop[j], G, nodes_attr, edges_attr)
                new_pop.append(C_cross)
        print("Finish Crossover.")

        # ==== Step 3: mutation ====
        p_mut = 0.5
        if iter == 1: p_mut = 1.0
        n = len(new_pop)

        for i in range(n):
            if p_mut >= rd.random():
                C_mutation = mutation(new_pop[i], G, important_nodes_idx, nodes_attr, edges_attr)
                new_pop.append(C_mutation)
        print("Finish Mutation.")

        # ==== update population ====
        pop = new_pop

        if members_best == pop[0].members:
            converge += 1
        else:
            converge = 0
            members_best = pop[0].members

        comm_size = len(pop[0].members)
        fitness_best = pop[0].fitness(nodes_attr)
        fitness_avg = np.mean([pop[i].fitness(nodes_attr) for i in range(len(pop))])

        comm_size_line.append(comm_size)
        fitness_best_line.append(fitness_best)
        fitness_avg_line.append(fitness_avg)
        print("Best Fitness: {}, Avg Fitness: {}, Comm Size: {}\n".format(fitness_best, fitness_avg, comm_size))

    # ==== save results ====
    save_comm_files(save_path, label_dict, phishing_label_set, pop[0], G, nodes_attr, edges_attr, idx2add)



if __name__ == '__main__':
    # ======== random seed ========
    rd.seed(42)

    # ======== data path ========
    parameters = open('params.json', 'r').read()
	parameters = json.loads(parameters)
    dataset_dir = parameters['data_path'] + '/APPR_alpha{}_epsilon{}/'.format(parameters['alpha'], parameters['epsilon'])
    motif_dir = parameters['motif_path'] + '/motif' + str(delta) + '/' + 'APPR_alpha{}_epsilon{}/'.format(parameters['alpha'], parameters['epsilon'])
    save_dir = "./out/"

    # ======== load labels ========
    life_dict = load_life_data()
    label_dict = load_label_data()
    phishing_label_set = load_phishing_label()

    # ========= community detection =========
    task_id = int((__file__.split("_")[-1]).replace(".py", ""))
    start = task_id * 200
    end = (task_id + 1) * 200 if task_id < 9 else 2056

    k = 0
    for seed in os.listdir(dataset_dir)[start: end]:
        k += 1

        subgraph_path = dataset_dir + seed + "/" + seed + ".csv"
        important_nodes_path = dataset_dir + seed + "/importance/" + seed + ".csv"
        motif_path = motif_dir + seed + ".npy"

        save_path = save_dir + str(task_id) + "/" + seed + "/"
        if not os.path.exists(save_path): os.makedirs(save_path)

        # load subgraph data
        etherdata = EtherData()
        etherdata.load_subgraph_data(seed, subgraph_path, important_nodes_path, motif_path, 'S5')

        # if the seed is isolated, directly save the community
        if etherdata.is_isolated():
            save_isolated_comm_files(save_path, seed, life_dict[seed])
            continue

        # load the lifetime data
        etherdata.load_life(life_dict)

        # genetic algorithm
        GA(pop_size=100, converge_num=30, seed=seed, etherdata=etherdata, save_path=save_path)
        del etherdata
        print("{} Finish Community Detection of seed {}.\n".format(k, seed))
