import pandas as pd
import numpy as np
import os

from sklearn.cluster import Birch
from sklearn.metrics.cluster import silhouette_score
from Levenshtein import jaro

import warnings
warnings.filterwarnings("ignore")



def load_scamdb_seeds():
    scamdb_df = pd.read_excel("data/APPR_scamdb_2056.xlsx")
    scamdb_seeds = list(scamdb_df["address"].values)

    print("load 2056 scamdb seeds\n")
    return scamdb_seeds


def load_scamdb_hostnames():
    hostname_df = pd.read_excel("data/labels_scamdb_hostnames.xlsx")
    hostname_dict = {}

    for i in range(hostname_df.shape[0]):
        addr, hostname = hostname_df.loc[i, ["address", "hostname"]]

        if addr not in hostname_dict:
            hostname_dict[addr] = [hostname]
        else:
            hostname_dict[addr].append(hostname)

    print("load hostnames of 2056 ScamDB address\n")
    return hostname_dict


def cal_overlap_matrix():
    M = np.zeros((n, n))

    comm_nodes_list = []

    for i in range(n):
        comm_nodes_df = pd.read_csv(comm_dir + "/output_GA/" + scamdb_seeds[i] + "/comm_nodes_inner.csv")
        comm_nodes = set(comm_nodes_df["Id"].values)
        comm_nodes_list.append(comm_nodes)

    for i in range(n):
        for j in range(i, n):
            M[i][j] = len(comm_nodes_list[i] & comm_nodes_list[j]) / min(len(comm_nodes_list[i]), len(comm_nodes_list[j]))
            M[j][i] = M[i][j]
            print(i, j, M[i][j])

    M_df = pd.DataFrame(M)
    M_df.to_csv(comm_dir + "/output_merge_with_clustering/overlap_matrix.csv", index=False)


def cal_similarity_matrix():
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            hostname_i = hostname_dict[scamdb_seeds[i]]
            hostname_j = hostname_dict[scamdb_seeds[j]]

            if len(hostname_i) == 1 and len(hostname_j) == 1:
                M[i][j] = jaro(hostname_i[0], hostname_j[0])

            else:
                M[i][j] = 0
                for k1 in range(len(hostname_i)):
                    for k2 in range(len(hostname_j)):
                        M[i][j] = max(M[i][j], jaro(hostname_i[k1], hostname_j[k2]))

            M[j][i] = M[i][j]
            print(i, j, M[i][j])

    M_df = pd.DataFrame(M)
    M_df.to_csv(comm_dir + "/output_merge_with_clustering/similarity_matrix.csv", index = False)



def merge_with_BIRCH():
    overlap_matrix_df = pd.read_csv(comm_dir + "/output_merge_with_clustering/overlap_matrix.csv")
    overlap_matrix = overlap_matrix_df.values
    similarity_matrix_df = pd.read_csv(comm_dir + "/output_merge_with_clustering/similarity_matrix.csv")
    similarity_matrix = similarity_matrix_df.values

    seed_labels = Birch(n_clusters = None).fit_predict(overlap_matrix)
    print("#gangs: {}".format(len(set(seed_labels))))
    score = silhouette_score(X = 1 - similarity_matrix, metric = "precomputed", labels = seed_labels)
    print("silhoutte score: {}".format(score))

    merge_comm_dir = comm_dir + "/output_merge_with_clustering/comm_merge_BIRCH/"
    if not os.path.exists(merge_comm_dir): os.makedirs(merge_comm_dir)

    gang_seeds_num = []
    gang_nodes_num = []
    gang_url_list = []
    gang_phish_num = []
    k = 0

    for label in set(seed_labels):
        gang_seeds_id = [i for i in range(len(seed_labels)) if seed_labels[i] == label]
        gang_seeds = [scamdb_seeds[id] for id in gang_seeds_id]
        gang_urls = [hostname_dict[seed] for seed in gang_seeds]

        mrg_comm_nodes_inner_df = pd.DataFrame()  # (only includes members within the community)
        mrg_comm_edges_inner_df = pd.DataFrame()
        mrg_comm_nodes_df = pd.DataFrame()  # (containing first-order neighbors of nodes outside the community)
        mrg_comm_edges_df = pd.DataFrame()

        for seed in gang_seeds:
            path_comm_nodes_inner = comm_dir + "/output_GA/" + seed + "/comm_nodes_inner.csv"
            path_comm_nodes = comm_dir + "/output_GA/" + seed + "/comm_nodes.csv"
            path_comm_edges_inner = comm_dir + "/output_GA/" + seed + "/comm_edges_inner.csv"
            path_comm_edges = comm_dir + "/output_GA/" + seed + "/comm_edges.csv"

            comm_nodes_inner_df = pd.read_csv(path_comm_nodes_inner)
            mrg_comm_nodes_inner_df = pd.concat([mrg_comm_nodes_inner_df, comm_nodes_inner_df], axis=0)

            if os.path.exists(path_comm_nodes):
                comm_nodes_df = pd.read_csv(path_comm_nodes)
                mrg_comm_nodes_df = pd.concat([mrg_comm_nodes_df, comm_nodes_df], axis=0)

            if os.path.exists(path_comm_edges_inner):
                comm_edges_inner_df = pd.read_csv(path_comm_edges_inner)
                mrg_comm_edges_inner_df = pd.concat([mrg_comm_edges_inner_df, comm_edges_inner_df])
                mrg_comm_edges_inner_df = mrg_comm_edges_inner_df.drop_duplicates()
                mrg_comm_edges_inner_df = mrg_comm_edges_inner_df.reset_index(drop=True)

            if os.path.exists(path_comm_edges):
                comm_edges_df = pd.read_csv(path_comm_edges)
                mrg_comm_edges_df = pd.concat([mrg_comm_edges_df, comm_edges_df])
                mrg_comm_edges_df = mrg_comm_edges_df.drop_duplicates()
                mrg_comm_edges_df = mrg_comm_edges_df.reset_index(drop=True)

        path = merge_comm_dir + "/gang_{}_{}/".format(k, len(gang_seeds))
        os.makedirs(path)

        mrg_comm_edges_inner_df["Id"] = np.arange(mrg_comm_edges_inner_df.shape[0])
        mrg_comm_edges_df["Id"] = np.arange(mrg_comm_edges_df.shape[0])

        mrg_comm_nodes_inner_df = mrg_comm_nodes_inner_df.groupby("Id", as_index=False).agg(np.max)
        if int(mrg_comm_nodes_df.shape[0]) > 0: mrg_comm_nodes_df = mrg_comm_nodes_df.groupby("Id", as_index=False).agg(np.max)

        mrg_comm_nodes_inner_df["seed_flag"] = np.array([1 if addr in set(gang_seeds) else 0 for addr in mrg_comm_nodes_inner_df["Id"].values])
        mrg_comm_nodes_inner_df["url"] = np.array([hostname_dict[addr] if addr in hostname_dict else "" for addr in mrg_comm_nodes_inner_df["Id"].values])

        if int(mrg_comm_nodes_df.shape[0]) > 0:
            mrg_comm_nodes_df["seed_flag"] = np.array([1 if addr in set(gang_seeds) else 0 for addr in mrg_comm_nodes_df["Id"].values])
            mrg_comm_nodes_df["url"] = np.array([hostname_dict[addr] if addr in hostname_dict else "" for addr in mrg_comm_nodes_df["Id"].values])

        mrg_comm_nodes_inner_df.to_csv(path + "mrg_comm_nodes_inner.csv", index=False)
        mrg_comm_nodes_df.to_csv(path + "mrg_comm_nodes.csv", index=False)
        mrg_comm_edges_inner_df.to_csv(path + "mrg_comm_edges_inner.csv", index=False)
        mrg_comm_edges_df.to_csv(path + "mrg_comm_edges.csv", index=False)

        gang_seeds_num.append(len(gang_seeds))
        gang_nodes_num.append(mrg_comm_nodes_inner_df.shape[0])
        gang_url_list.append(gang_urls)
        gang_phish_num.append(mrg_comm_nodes_inner_df[mrg_comm_nodes_inner_df["Label"] == "phish-hack"].shape[0])

        k += 1

    df = pd.DataFrame({
        "gang_id": [i for i in range(len(gang_seeds_num))],
        "seed_num": gang_seeds_num,
        "node_num": gang_nodes_num,
        "phish_num": gang_phish_num,
        "url_list": gang_url_list
    })
    df.to_csv(comm_dir + "/output_merge_with_clustering/gang_info.csv", index=False)



if __name__ == '__main__':
    comm_dir = "./results"
    if not os.path.exists(comm_dir + "/output_merge_with_clustering/"): os.makedirs(comm_dir + "/output_merge_with_clustering/")

    scamdb_seeds = load_scamdb_seeds()
    hostname_dict = load_scamdb_hostnames()
    n = len(scamdb_seeds)

    if not os.path.exists(comm_dir + "/output_merge_with_clustering/overlap_matrix.csv"):
        cal_overlap_matrix()

    if not os.path.exists(comm_dir + "/output_merge_with_clustering/similarity_matrix.csv"):
        cal_similarity_matrix()

    # clustering
    merge_with_BIRCH()
