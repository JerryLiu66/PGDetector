import matplotlib.pyplot as plt
import pandas as pd



def load_label_data():
    label_df = pd.read_csv("data/labels_etherscan_33w.csv")
    label_dict = label_df.set_index(["Address"])["Label_1"].to_dict()
    print("load labels_etherscan_33w.csv\n")
    return label_dict


def load_phishing_label():
    # 5362 phishing labels
    phish_label_df = pd.read_csv("data/phishing_label_5362.csv")
    phish_label_set = set(phish_label_df["address"].values)
    print("load phishing_label_5362.csv\n")
    return phish_label_set


def load_life_data():
	import json
	parameters = open('params.json', 'r').read()
	parameters = json.loads(parameters)
    life_df = pd.read_csv("data/lifetime/APPR_alpha{}_epsilon{}/scamdb_2056_life.csv".format(parameters['alpha'], parameters['epsilon']))
    life_dict = life_df.set_index(["address"])["life"].to_dict()
    print("load scamdb_2056_life.csv\n")
    return life_dict


def draw_figures(save_path, comm_size_line, fitness_best_line, fitness_avg_line):
    # Community Size
    plt.plot(comm_size_line)
    plt.xlabel("Iteration");  plt.ylabel("Community Size")
    plt.savefig(save_path + "Comm_Size.png")
    plt.close()

    # Fitness
    plt.figure(figsize = (8, 6))
    plt.plot(fitness_best_line, label = "Best Fitness")
    plt.plot(fitness_avg_line, label = "Avg Fitness")
    plt.xlabel("Iteration");  plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(save_path + "Fitness.png")
    plt.close()


def save_isolated_comm_files(save_path, seed, seed_life):
    comm_nodes_df = pd.DataFrame({
        "Id": [seed],
        "Life": [seed_life],
        "Label": ["phish-hack"]
    })
    comm_nodes_df.to_csv(save_path + "comm_nodes_inner.csv", index = False)


def save_comm_files(save_path, label_dict, phish_label_set, C, G, nodes_attr, edges_attr, idx2add):
    """
    comm_nodes.csv: Node file (containing first-order neighbors of nodes outside the community)
    comm_edges.csv: Edge file (containing first-order neighbors of nodes outside the community)
    comm_nodes_inner.csv: Node file (only includes members within the community)
    comm_edges_inner.csv: Edge file (only includes members within the community)
    """
    src_list, trg_list, weight_list, value_list, cnt_list = [], [], [], [], []
    src_list_inner, trg_list_inner, weight_list_inner, value_list_inner, cnt_list_inner = [], [], [], [], []
    nodes_idx_set = set()

    for idx in C.members:
        for k in G.incident(idx, mode="all"):
            src = G.es[k].source
            trg = G.es[k].target

            nodes_idx_set.add(src)
            nodes_idx_set.add(trg)

            src_list.append(idx2add[src])
            trg_list.append(idx2add[trg])
            weight_list.append(edges_attr[(src, trg)]["weight"])
            value_list.append(edges_attr[(src, trg)]["value"])
            cnt_list.append(edges_attr[(src, trg)]["cnt"])

            if (src in C.members) and (trg in C.members):
                src_list_inner.append(idx2add[src])
                trg_list_inner.append(idx2add[trg])
                weight_list_inner.append(edges_attr[(src, trg)]["weight"])
                value_list_inner.append(edges_attr[(src, trg)]["value"])
                cnt_list_inner.append(edges_attr[(src, trg)]["cnt"])

    id_list = [i for i in range(len(src_list))]
    id_list_inner = [i for i in range(len(src_list_inner))]

    # (containing first-order neighbors of nodes outside the community)
    comm_edges_df = pd.DataFrame({
        "Id": id_list,
        "Source": src_list,
        "Target": trg_list,
        "Weight": weight_list,
        "Value": value_list,
        "Cnt": cnt_list
    })
    comm_edges_df.to_csv(save_path + "comm_edges.csv", index = False)

    # (only includes members within the community)
    comm_edges_df = pd.DataFrame({
        "Id": id_list_inner,
        "Source": src_list_inner,
        "Target": trg_list_inner,
        "Weight": weight_list_inner,
        "Value": value_list_inner,
        "Cnt": cnt_list_inner
    })
    comm_edges_df.to_csv(save_path + "comm_edges_inner.csv", index = False)

    # ======== save nodes ========

    # (containing first-order neighbors of nodes outside the community)
    nodes_idx_list = list(nodes_idx_set)
    nodes_list = [idx2add[idx] for idx in nodes_idx_list]
    life_list = []
    label_list = []
    gang_member_flag_list = [1 if idx in C.members else 0 for idx in nodes_idx_list]

    for idx in nodes_idx_list:
        if "life" in nodes_attr[idx]: life_list.append(nodes_attr[idx]["life"])
        else: life_list.append("")

    for addr in nodes_list:
        if addr in label_dict: label_list.append(label_dict[addr])
        elif addr in phish_label_set: label_list.append("phish-hack")
        else: label_list.append("")

    comm_nodes_df = pd.DataFrame({
        "Id": nodes_list,
        "Life": life_list,
        "Label": label_list,
        "gang_member_flag": gang_member_flag_list
    })
    comm_nodes_df.to_csv(save_path + "comm_nodes.csv", index = False)

    # (only includes members within the community)
    nodes_idx_list_inner = list(C.members)
    nodes_list_inner = [idx2add[idx] for idx in nodes_idx_list_inner]
    life_list_inner = [nodes_attr[idx]["life"] for idx in nodes_idx_list_inner]
    label_list_inner = []

    for addr in nodes_list_inner:
        if addr in label_dict: label_list_inner.append(label_dict[addr])
        elif addr in phish_label_set: label_list_inner.append("phish-hack")
        else: label_list_inner.append("")

    comm_nodes_df = pd.DataFrame({
        "Id": nodes_list_inner,
        "Life": life_list_inner,
        "Label": label_list_inner
    })
    comm_nodes_df.to_csv(save_path + "comm_nodes_inner.csv", index=False)
