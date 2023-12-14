import pandas as pd
import numpy as np
import os
import shutil
import json



def load_label_data(): # non-phishing accounts
    label_df = pd.read_csv("data/labels_etherscan_33w.csv")
    label_dict = label_df.set_index(["Address"])["Label_1"].to_dict()
    print("load labels_etherscan_33w.csv\n")
    return label_dict


def load_phishing_label():
	parameters = open('params.json', 'r').read()
	parameters = json.loads(parameters)
    phish_label_df = pd.read_csv("data/phishing_label_5362.csv")
    phish_label_set = set(phish_label_df["address"].values)
    print("load phishing_label_5362.csv\n")

    scamdb_seeds = set(os.listdir(parameters['data_path'] + '/APPR_alpha{}_epsilon{}/'.format(parameters['alpha'], parameters['epsilon'])))
    print("load 2056 scamdb seeds\n")
    return phish_label_set, scamdb_seeds


def evaluate_results(comm_dir):
    """
    Evaluation
    """
    detected_address = set()
    TP_address = set()
    FP_address = set()
    exchange_address = set()

    # ==== Statistics ====
    label_dict = load_label_data()
    phish_label_set, scamdb_seeds = load_phishing_label()

    for seed in os.listdir(comm_dir + "/output_GA/"):
        with open(comm_dir + "/output_GA/" + seed + "/comm_nodes_inner.csv", "r") as f:
            for line in f:
                line = line.rstrip("\n").split(",")
                if line[0] == "Id": continue

                addr = line[0]
                detected_address.add(addr)
                if addr in (phish_label_set - scamdb_seeds): TP_address.add(addr)
                if (addr in label_dict) and (label_dict[addr] != "phish-hack"): FP_address.add(addr)
                if (addr in label_dict) and (label_dict[addr] == "exchange"): exchange_address.add(addr)

    print("#Nodes: {}".format(len(detected_address)))
    print("TP: {}".format(len(TP_address)))
    print("FP: {}".format(len(FP_address)))
    print("#exchanges: {}".format(len(exchange_address)))

    avg_phish_density = cal_avg_phish_density(comm_dir, phish_label_set, scamdb_seeds)
    print("avg_phish_density: {}".format(avg_phish_density))

    avg_conductance = cal_avg_conductance(comm_dir)
    print("avg_conductance: {}".format(avg_conductance))

    avg_weighted_conductance = cal_avg_weighted_conductance(comm_dir)
    print("avg_weighted_conductance: {}".format(avg_weighted_conductance))



def cal_avg_phish_density(comm_dir, phish_label_set, scamdb_seeds):
    merge_dir = comm_dir + "/output_merge_with_clustering/comm_merge_BIRCH/"
    density_list = []

    for gang in os.listdir(merge_dir):
        df = pd.read_csv(merge_dir + gang + "/mrg_comm_nodes_inner.csv")
        num_phish = float(df[df["Id"].isin(phish_label_set | scamdb_seeds)].shape[0])
        num_nodes = float(df.shape[0])
        density_list.append(num_phish / num_nodes)

    return np.mean(density_list)


def cal_avg_conductance(comm_dir):
    merge_dir = comm_dir + "/output_merge_with_clustering/comm_merge_BIRCH/"
    conductance_list = []

    for gang in os.listdir(merge_dir):
        comm_nodes_inner_df = pd.read_csv(merge_dir + gang + "/mrg_comm_nodes_inner.csv")
        comm_edges_df = pd.read_csv(merge_dir + gang + "/mrg_comm_edges.csv")

        # For isolated seeds
        if int(comm_nodes_inner_df.shape[0]) == 1:
            conductance_list.append(1.0)
            continue

        # For others
        comm_nodes_inner = set(comm_nodes_inner_df["Id"].values)
        outer = inner = 0.0

        for i in range(comm_edges_df.shape[0]):
            src, trg = comm_edges_df.loc[i, ["Source", "Target"]]

            if src in comm_nodes_inner: inner += 1
            else: outer += 1
            if trg in comm_nodes_inner: inner += 1
            else: outer += 1

        conductance_list.append(outer / inner)

    return np.mean(conductance_list)


def cal_avg_weighted_conductance(comm_dir):
    merge_dir = comm_dir + "/output_merge_with_clustering/comm_merge_BIRCH/"
    weighted_conductance_list = []

    for gang in os.listdir(merge_dir):
        comm_nodes_inner_df = pd.read_csv(merge_dir + gang + "/mrg_comm_nodes_inner.csv")
        comm_edges_df = pd.read_csv(merge_dir + gang + "/mrg_comm_edges.csv")

        # For isolated seeds
        if int(comm_nodes_inner_df.shape[0]) == 1:
            weighted_conductance_list.append(1.0)
            continue

        # For others
        comm_nodes_inner = set(comm_nodes_inner_df["Id"].values)
        outer = inner = 0.0

        for i in range(comm_edges_df.shape[0]):
            src, trg = comm_edges_df.loc[i, ["Source", "Target"]]
            value = float(comm_edges_df.loc[i, "Value"])

            if src in comm_nodes_inner: inner += value
            else: outer += value
            if trg in comm_nodes_inner: inner += value
            else: outer += value

        weighted_conductance_list.append(outer / inner)

    return np.mean(weighted_conductance_list)



if __name__ == '__main__':
    comm_dir = "./results"
    evaluate_results(comm_dir)
