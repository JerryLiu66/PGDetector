import pandas as pd
import numpy as np
import heapq
import igraph as ig

from sklearn.preprocessing import RobustScaler



class EtherData(object):
    def __init__(self):
        self.seed = None
        self.G = None
        self.important_nodes = []
        self.important_nodes_idx = []

        self.add2idx = {}  # add2idx[addr] = idx
        self.idx2add = {}  # idx2add[idx] = addr
        self.nodes_attr = {}  # nodes_attr[idx] = {...}
        self.edges_attr = {}  # edges_attr[idx] = {...}


    def load_subgraph_data(self, seed, subgraph_path, important_nodes_path, motif_path, strategy='S5'):
        self.seed = seed

        important_nodes_df = pd.read_csv(important_nodes_path)
        self.important_nodes = list(important_nodes_df["node"].values)

        idx = 0
        V = set()
        E = set()

        with open(subgraph_path, "r") as f:
            for line in f:
                line = line.rstrip("\n").split(",")
                if line[0] == "hash": continue

                # delete self loop
                if line[1] == line[2]: continue
                # delete failed transaction
                if int(line[8]) == 1: continue
                value = float(line[3]) / 1e18

                if line[0] not in E:
                    E.add(line[0])

                    if line[1] not in V:
                        self.add2idx[line[1]] = idx
                        self.idx2add[idx] = line[1]
                        self.nodes_attr[idx] = {"sum_in_value": 0, "sum_out_value": value, "sum_in_cnt": 0, "sum_out_cnt": 1}
                        idx += 1
                        V.add(line[1])
                    else:
                        self.nodes_attr[self.add2idx[line[1]]]["sum_out_value"] += value
                        self.nodes_attr[self.add2idx[line[1]]]["sum_out_cnt"] += 1

                    if line[2] not in V:
                        self.add2idx[line[2]] = idx
                        self.idx2add[idx] = line[2]
                        self.nodes_attr[idx] = {"sum_in_value": value, "sum_out_value": 0, "sum_in_cnt": 1, "sum_out_cnt": 0}
                        idx += 1
                        V.add(line[2])
                    else:
                        self.nodes_attr[self.add2idx[line[2]]]["sum_in_value"] += value
                        self.nodes_attr[self.add2idx[line[2]]]["sum_in_cnt"] += 1

                    src_idx = self.add2idx[line[1]]
                    trg_idx = self.add2idx[line[2]]

                    if (src_idx, trg_idx) not in self.edges_attr:
                        self.edges_attr[(src_idx, trg_idx)] = {"value": value, "cnt": 1}
                    else:
                        self.edges_attr[(src_idx, trg_idx)]["value"] += value
                        self.edges_attr[(src_idx, trg_idx)]["cnt"] += 1

        vertices_list = sorted(list(self.nodes_attr.keys()))
        edges_list = list(self.edges_attr.keys())

        self.G = ig.Graph(directed = True)
        self.G.add_vertices(vertices_list)
        self.G.add_edges(edges_list)

        self.important_nodes = list(set(self.important_nodes) & V)  # 剔除失败/自环交易涉及的重要性节点
        self.important_nodes_idx = [self.add2idx[addr] for addr in self.important_nodes]

        # ==== edge weight ====
        assert strategy in ['S1', 'S2', 'S3', 'S4', 'S5']
        if strategy == 'S1':
            self.cal_edges_weight_S1()
        elif strategy == 'S2':
            self.cal_edges_weight_S2()
        elif strategy == 'S3':
            self.cal_edges_weight_S3(motif_path)
        elif strategy == 'S4':
            self.cal_edges_weight_S4()
        elif strategy == 'S5':
            self.cal_edges_weight_S5(motif_path)

        for idx in self.important_nodes_idx:
            self.nodes_attr[idx]["sum_weight"] = 0

            for src_idx in self.G.predecessors(idx):
                self.nodes_attr[idx]["sum_weight"] += self.edges_attr[(src_idx, idx)]["weight"]
            for trg_idx in self.G.successors(idx):
                self.nodes_attr[idx]["sum_weight"] += self.edges_attr[(idx, trg_idx)]["weight"]

        print("finish subgraph construction of seed {}, with {} nodes, {} edges\n".format(self.seed, self.G.vcount(), self.G.vcount()))


    def load_life(self, life_dict):
        """
        加载重要性节点的生命周期, 并标准化
        """
        sum_life = 0

        for addr in self.important_nodes:
            self.nodes_attr[self.add2idx[addr]]["life"] = life_dict[addr]
            sum_life += life_dict[addr]

        for addr in self.important_nodes:
            self.nodes_attr[self.add2idx[addr]]["life_norm"] = life_dict[addr] / sum_life


    def normalize_edges_rate(self, rateType):
        for type in rateType:
            rateList = [attr[type] for e, attr in self.edges_attr.items()]

            scaler = RobustScaler(with_centering=True, with_scaling=True)
            rateList = scaler.fit_transform(np.array(rateList).reshape(-1, 1))[:, 0].tolist()

            rateList_min = min(rateList)
            for i in range(len(rateList)):  rateList[i] = rateList[i] - rateList_min + 1
            rateList_max = max(rateList)
            for i in range(len(rateList)):  rateList[i] = np.log10(rateList[i]) / np.log10(rateList_max)

            i = 0
            for e, attr in self.edges_attr.items():
                self.edges_attr[e][type + "_norm"] = rateList[i]
                i += 1


    def cal_edges_weight_S1(self):
        for e, attr in self.edges_attr.items():
            src_idx, trg_idx = e
            vij = attr["value"]

            vi_out = self.nodes_attr[src_idx]["sum_out_value"]
            vj_in = self.nodes_attr[trg_idx]["sum_in_value"]

            v_in_rate = vij / vj_in if vj_in != 0 else 0
            v_out_rate = vij / vi_out if vi_out != 0 else 0

            self.edges_attr[(src_idx, trg_idx)]["v_in_rate"] = v_in_rate
            self.edges_attr[(src_idx, trg_idx)]["v_out_rate"] = v_out_rate

        self.normalize_edges_rate(["v_in_rate", "v_out_rate"])

        for e, attr in self.edges_attr.items():
            self.edges_attr[e]["weight"] = attr["value"] * max(
                attr["v_in_rate_norm"], attr["v_out_rate_norm"]
            )


    def cal_edges_weight_S2(self):
        for e, attr in self.edges_attr.items():
            src_idx, trg_idx = e
            cij = attr["cnt"]

            ci_out = self.nodes_attr[src_idx]["sum_out_cnt"]
            cj_in = self.nodes_attr[trg_idx]["sum_in_cnt"]

            c_in_rate = cij / cj_in if cj_in != 0 else 0
            c_out_rate = cij / ci_out if ci_out != 0 else 0

            self.edges_attr[(src_idx, trg_idx)]["c_in_rate"] = c_in_rate
            self.edges_attr[(src_idx, trg_idx)]["c_out_rate"] = c_out_rate

        self.normalize_edges_rate(["c_in_rate", "c_out_rate"])

        for e, attr in self.edges_attr.items():
            self.edges_attr[e]["weight"] = attr["value"] * max(
                attr["c_in_rate_norm"], attr["c_out_rate_norm"]
            )


    def cal_edges_weight_S3(self, motif_path):
        motif_matrix = np.load(motif_path, allow_pickle=True).item()
        motif_matrix_sum = {addr: sum(list(motif_matrix[addr].values())) for addr in motif_matrix}

        for e, attr in self.edges_attr.items():
            src_idx, trg_idx = e
            cij = attr["cnt"]

            ci_out = self.nodes_attr[src_idx]["sum_out_cnt"]
            cj_in = self.nodes_attr[trg_idx]["sum_in_cnt"]

            src = self.idx2add[src_idx]
            trg = self.idx2add[trg_idx]
            motif_c = 0
            motif_c_sum = 0

            if src in motif_matrix:
                motif_c_sum = motif_matrix_sum[src]
                if trg in motif_matrix[src]: motif_c = motif_matrix[src][trg]

            cj_in += motif_c_sum
            ci_out += motif_c_sum
            cij += motif_c

            c_in_rate = cij / cj_in if cj_in != 0 else 0
            c_out_rate = cij / ci_out if ci_out != 0 else 0

            self.edges_attr[(src_idx, trg_idx)]["c_in_rate"] = c_in_rate
            self.edges_attr[(src_idx, trg_idx)]["c_out_rate"] = c_out_rate

        self.normalize_edges_rate(["c_in_rate", "c_out_rate"])

        for e, attr in self.edges_attr.items():
            self.edges_attr[e]["weight"] = attr["value"] * max(
                attr["c_in_rate_norm"], attr["c_out_rate_norm"]
            )


    def cal_edges_weight_S4(self):
        for e, attr in self.edges_attr.items():
            src_idx, trg_idx = e
            vij = attr["value"]
            cij = attr["cnt"]

            vi_out = self.nodes_attr[src_idx]["sum_out_value"]
            ci_out = self.nodes_attr[src_idx]["sum_out_cnt"]
            vj_in = self.nodes_attr[trg_idx]["sum_in_value"]
            cj_in = self.nodes_attr[trg_idx]["sum_in_cnt"]

            v_in_rate = vij / vj_in if vj_in != 0 else 0
            v_out_rate = vij / vi_out if vi_out != 0 else 0

            c_in_rate = cij / cj_in if cj_in != 0 else 0
            c_out_rate = cij / ci_out if ci_out != 0 else 0

            self.edges_attr[(src_idx, trg_idx)]["v_in_rate"] = v_in_rate
            self.edges_attr[(src_idx, trg_idx)]["v_out_rate"] = v_out_rate
            self.edges_attr[(src_idx, trg_idx)]["c_in_rate"] = c_in_rate
            self.edges_attr[(src_idx, trg_idx)]["c_out_rate"] = c_out_rate

        self.normalize_edges_rate(["v_in_rate", "v_out_rate", "c_in_rate", "c_out_rate"])

        for e, attr in self.edges_attr.items():
            self.edges_attr[e]["weight"] = attr["value"] * 1/2 * max(
                attr["v_in_rate_norm"] + attr["c_in_rate_norm"], attr["v_out_rate_norm"] + attr["c_out_rate_norm"]
            )


    def cal_edges_weight_S5(self, motif_path):
        motif_matrix = np.load(motif_path, allow_pickle=True).item()
        motif_matrix_sum = {addr: sum(list(motif_matrix[addr].values())) for addr in motif_matrix}

        for e, attr in self.edges_attr.items():
            src_idx, trg_idx = e
            vij = attr["value"]
            cij = attr["cnt"]

            vi_out = self.nodes_attr[src_idx]["sum_out_value"]
            ci_out = self.nodes_attr[src_idx]["sum_out_cnt"]
            vj_in = self.nodes_attr[trg_idx]["sum_in_value"]
            cj_in = self.nodes_attr[trg_idx]["sum_in_cnt"]

            v_in_rate = vij / vj_in if vj_in != 0 else 0
            v_out_rate = vij / vi_out if vi_out != 0 else 0

            src = self.idx2add[src_idx]
            trg = self.idx2add[trg_idx]
            motif_c = 0
            motif_c_sum = 0

            if src in motif_matrix:
                motif_c_sum = motif_matrix_sum[src]
                if trg in motif_matrix[src]: motif_c = motif_matrix[src][trg]

            cj_in += motif_c_sum
            ci_out += motif_c_sum
            cij += motif_c

            c_in_rate = cij / cj_in if cj_in != 0 else 0
            c_out_rate = cij / ci_out if ci_out != 0 else 0

            self.edges_attr[(src_idx, trg_idx)]["v_in_rate"] = v_in_rate
            self.edges_attr[(src_idx, trg_idx)]["v_out_rate"] = v_out_rate
            self.edges_attr[(src_idx, trg_idx)]["c_in_rate"] = c_in_rate
            self.edges_attr[(src_idx, trg_idx)]["c_out_rate"] = c_out_rate

        self.normalize_edges_rate(["v_in_rate", "v_out_rate", "c_in_rate", "c_out_rate"])

        for e, attr in self.edges_attr.items():
            self.edges_attr[e]["weight"] = attr["value"] * 1/2 * max(
                attr["v_in_rate_norm"] + attr["c_in_rate_norm"], attr["v_out_rate_norm"] + attr["c_out_rate_norm"]
            )


    def get_nodes_info(self, out_path, label_dict, phish_label_set):
        label_list = []

        for addr in self.important_nodes:
            if addr in label_dict: label_list.append(label_dict[addr])
            elif addr in phish_label_set: label_list.append("phish-hack")
            else: label_list.append("")

        nodes_info_df = pd.DataFrame({
            "address": self.important_nodes,
            "sum_in_value": [self.nodes_attr[idx]["sum_in_value"] for idx in self.important_nodes_idx],
            "sum_in_cnt": [self.nodes_attr[idx]["sum_in_cnt"] for idx in self.important_nodes_idx],
            "sum_out_value": [self.nodes_attr[idx]["sum_out_value"] for idx in self.important_nodes_idx],
            "sum_out_cnt": [self.nodes_attr[idx]["sum_out_cnt"] for idx in self.important_nodes_idx],
            "sum_weight": [self.nodes_attr[idx]["sum_weight"] for idx in self.important_nodes_idx],
            "life": [self.nodes_attr[idx]["life"] for idx in self.important_nodes_idx],
            "life_norm": [self.nodes_attr[idx]["life_norm"] for idx in self.important_nodes_idx],
            "label": label_list
        })
        nodes_info_df.to_csv(out_path, index = False)


    def get_edges_info(self, out_path):
        from_list, to_list, value_list, cnt_list, weight_list = [], [], [], [], []
        v_in_rate_list, c_in_rate_list, v_out_rate_list, c_out_rate_list = [], [], [], []
        v_in_rate_norm_list, c_in_rate_norm_list, v_out_rate_norm_list, c_out_rate_norm_list = [], [], [], []

        for e, attr in self.edges_attr.items():
            from_list.append(self.idx2add[e[0]])
            to_list.append(self.idx2add[e[1]])
            value_list.append(attr["value"])
            cnt_list.append(attr["cnt"])
            weight_list.append(attr["weight"])

            v_in_rate_list.append(attr["v_in_rate"])
            c_in_rate_list.append(attr["c_in_rate"])
            v_out_rate_list.append(attr["v_out_rate"])
            c_out_rate_list.append(attr["c_out_rate"])

            v_in_rate_norm_list.append(attr["v_in_rate_norm"])
            c_in_rate_norm_list.append(attr["c_in_rate_norm"])
            v_out_rate_norm_list.append(attr["v_out_rate_norm"])
            c_out_rate_norm_list.append(attr["c_out_rate_norm"])

        edges_info_df = pd.DataFrame({
            "from": from_list,
            "to": to_list,
            "value": value_list,
            "cnt": cnt_list,
            "weight": weight_list,

            "v_in_rate": v_in_rate_list,
            "c_in_rate": c_in_rate_list,
            "v_out_rate": v_out_rate_list,
            "c_out_rate": c_out_rate_list,

            "v_in_rate_norm": v_in_rate_norm_list,
            "c_in_rate_norm": c_in_rate_norm_list,
            "v_out_rate_norm": v_out_rate_norm_list,
            "c_out_rate_norm": c_out_rate_norm_list
        })
        edges_info_df.to_csv(out_path, index = False)


    def is_isolated(self):
        seed_idx = self.add2idx[self.seed]
        return (self.nodes_attr[seed_idx]["sum_weight"] == 0) or (len(self.important_nodes) == 1)



class Comm(object):
    def __init__(self):
        self.seed_idx = None
        self.members = set()
        self.depth = {}
        self.inner = self.outer = 0


    def init_comm(self, seed_idx, nodes_attr):
        self.seed_idx = seed_idx
        self.members.add(seed_idx)
        self.depth[seed_idx] = 0
        self.inner = self.outer = nodes_attr[seed_idx]["sum_weight"]


    def push_node(self, idx, dp, G, nodes_attr, edges_attr):
        for k in G.incident(idx, mode = "all"):
            src_idx = G.es[k].source
            trg_idx = G.es[k].target

            if src_idx == idx: neigh_idx = trg_idx
            elif trg_idx == idx: neigh_idx = src_idx

            if neigh_idx in self.members:
                self.outer -= edges_attr[(src_idx, trg_idx)]["weight"]
            else:
                self.outer += edges_attr[(src_idx, trg_idx)]["weight"]
        self.inner += nodes_attr[idx]["sum_weight"]
        self.depth[idx] = dp
        self.members.add(idx)


    def pop_node(self, idx, G, nodes_attr, edges_attr):
        for k in G.incident(idx, mode = "all"):
            src_idx = G.es[k].source
            trg_idx = G.es[k].target

            if src_idx == idx:
                neigh_idx = trg_idx
            elif trg_idx == idx:
                neigh_idx = src_idx

            if neigh_idx in self.members:
                self.outer += edges_attr[(src_idx, trg_idx)]["weight"]
            else:
                self.outer -= edges_attr[(src_idx, trg_idx)]["weight"]
        self.inner -= nodes_attr[idx]["sum_weight"]
        self.depth.pop(idx)
        self.members.remove(idx)


    def deepest_members(self):
        return [idx for idx, dp in self.depth.items() if dp == heapq.nlargest(1, self.depth.values())[0]]


    def fitness(self, nodes_attr):
        cond = self.outer / self.inner
        avg_life = sum([nodes_attr[idx]["life_norm"] for idx in self.members]) / len(self.members)  # 惩罚项

        alpha = 1e2
        return cond + alpha * avg_life
