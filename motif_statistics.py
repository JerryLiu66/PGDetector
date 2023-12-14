import pandas as pd
import numpy as np
import os
import json

from numba.typed import List, Dict
from numba import njit, types
import numba
import json


@numba.njit(cache = True, locals = {'cnt_phish': numba.int64})
def motif_cnt(node, parents, children, motif_matrix, delta):

    if node in parents and node in children:
        for child in children[node]:
            for parent in parents[node]:

                # abba 1
                if child[0] != parent[0] and parent[1] <= child[1] <= parent[1] + delta:

                    if node in motif_matrix:
                        motif_matrix[node].append(child[0])
                        motif_matrix[node].append(parent[0])
                    else:
                        L = List()
                        L.append(child[0])
                        L.append(parent[0])
                        motif_matrix[node] = L
                        # motif_matrix[node] = [child[0], parent[0]]

                    if child[0] in motif_matrix:
                        motif_matrix[child[0]].append(node)
                        motif_matrix[child[0]].append(parent[0])
                    else:
                        L = List()
                        L.append(node)
                        L.append(parent[0])
                        motif_matrix[child[0]] = L
                        # motif_matrix[child[0]] = [node, parent[0]]

                    if parent[0] in motif_matrix:
                        motif_matrix[parent[0]].append(node)
                        motif_matrix[parent[0]].append(child[0])
                    else:
                        L = List()
                        L.append(node)
                        L.append(child[0])
                        motif_matrix[parent[0]] = L
                        # motif_matrix[parent[0]] = [node, child[0]]

                    # print(len(motif_matrix), len(motif_matrix[node]))

                elif parent[1] > child[1]:
                    break


@numba.njit(cache=True)
def cal_motif(parents, children, delta, motif_matrix, nodes):
    for k, node in enumerate(nodes):
        motif_cnt(node, parents, children, motif_matrix, delta)


def process_subgraph(datasets_dir, seed):
    subgraph_df = pd.read_csv(datasets_dir + seed + "/" + seed + ".csv", low_memory = False)
    appr_node_df = pd.read_csv(datasets_dir + seed + "/importance/" + seed + ".csv")

    subgraph_df = subgraph_df.drop_duplicates(subset = ["hash"])
    subgraph_df = subgraph_df.reset_index(drop = True)

    del_idx = set()
    del_idx |= set(subgraph_df[subgraph_df["from"] == subgraph_df["to"]].index)
    del_idx |= set(subgraph_df[subgraph_df["isError"] == 1].index)

    subgraph_df = subgraph_df.drop(del_idx)
    subgraph_df = subgraph_df.reset_index(drop = True)

    parents, children = {}, {}
    add2idx, idx2add = {}, {}
    idx = numba.int64(0)

    for i in range(subgraph_df.shape[0]):
        hash, src, trg = subgraph_df.loc[i, ["hash", "from", "to"]]
        timeStamp = numba.int64(subgraph_df.loc[i, "timeStamp"])

        if src not in add2idx:
            add2idx[src] = idx
            idx2add[idx] = src
            idx += 1
        if trg not in add2idx:
            add2idx[trg] = idx
            idx2add[idx] = trg
            idx += 1

        idx_src = add2idx[src]
        idx_trg = add2idx[trg]

        if src != trg:
            if idx_src not in children:
                children[idx_src] = [[idx_trg, timeStamp]]
            else:
                children[idx_src].append([idx_trg, timeStamp])

            if idx_trg not in parents:
                parents[idx_trg] = [[idx_src, timeStamp]]
            else:
                parents[idx_trg].append([idx_src, timeStamp])



    for key in parents:
        parents[key] = sorted(parents[key], key = lambda x: x[1], reverse = True)
    for key in children:
        children[key] = sorted(children[key], key = lambda x: x[1], reverse = True)


    P = Dict.empty(
        key_type = types.int64,
        value_type = types.ListType(types.ListType(types.int64))
    )
    C = Dict.empty(
        key_type = types.int64,
        value_type = types.ListType(types.ListType(types.int64))
    )

    for key, value in parents.items():
        L = List()
        for item in value:  L.append(List(item))
        P[key] = L

    for key, value in children.items():
        L = List()
        for item in value:  L.append(List(item))
        C[key] = L

    print("finish graph construction of seed {}".format(seed))
    return P, C, add2idx, idx2add


def feature_extraction(datasets_dir, save_path, delta):
    datasets = os.listdir(datasets_dir)

    for seed in datasets:
        parents, children, add2idx, idx2add = process_subgraph(datasets_dir, seed)
        nodes = np.array(list(set(parents.keys()) | set(children.keys())), dtype = np.int64)

        motif_matrix = Dict.empty(
            key_type = types.int64,
            value_type = types.ListType(types.int64) 
        )
        cal_motif(parents, children, delta, motif_matrix, nodes)

        
        motif_matrix_dict = {}

        for node1 in motif_matrix:
            for node2 in motif_matrix[node1]:

                # print(idx2add[node1], idx2add[node2], motif_matrix_dict)

                if idx2add[node1] not in motif_matrix_dict:
                    motif_matrix_dict[idx2add[node1]] = {idx2add[node2]: 1}
                elif idx2add[node2] not in motif_matrix_dict[idx2add[node1]]:
                    motif_matrix_dict[idx2add[node1]][idx2add[node2]] = 1
                else:
                    motif_matrix_dict[idx2add[node1]][idx2add[node2]] += 1

        np.save(save_path + seed + ".npy", motif_matrix_dict)



if __name__ == '__main__':

	parameters = open('params.json', 'r').read()
	parameters = json.loads(parameters)

    delta = parameters['motif_delta']
    save_path = parameters['motif_path'] + '/motif' + str(delta) + '/' + 'APPR_alpha{}_epsilon{}/'.format(parameters['alpha'], parameters['epsilon'])
    datasets_dir = parameters['data_path'] + '/APPR_alpha{}_epsilon{}/'.format(parameters['alpha'], parameters['epsilon'])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    feature_extraction(datasets_dir, save_path, delta)
