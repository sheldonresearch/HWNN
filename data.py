#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:HWNN
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: data.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2019/10/16
"""
import numpy as np
from collections import defaultdict
import torch


class Data:
    def __init__(self, metapathscheme=None):
        self.data_path = None
        self.nodes_labels = None  # one-hot torch matrix
        self.nodes_names_int = None  # list, int
        self.nodes_names_map = None  # content和cite节点的顺序不一样，为此通过这个进行映射查找
        self.X_0 = None
        self.class_num = 0  # first initialed in def _label_string2matrix
        self.nodes_labels_sequence = None  # node labels like [1,2,5,3,0]. please note the differece with self.nodes_labels
        self.edges = None  # simple graph edges (src,des,edge_type),
        self.nodes_number = 0
        self.s = 1.0
        self.hypergraph_snapshot = []
        self.simplegraph_snapshot = []
        self.metapathscheme = metapathscheme  # [[0,0]]
        self.labeled_node_index = []  # only valid for imdb dataset
        """
        We set self.hypergraph_snapshot[0], 
        and self.simplegraph_snapshot[0] as the compelete hypergraph/simple graph
        by default.
        
        for cora dataset, it is actually a homogeneous hypergraph. here we just copy the
        hypergraph three times, and generate three same hypergraph snapshots.
        """

    def _label_string2matrix(self, nodes_labels_str):
        b, c = np.unique(nodes_labels_str, return_inverse=True)
        class_num = len(b)
        sample_num = len(c)
        self.class_num = class_num
        self.nodes_labels_sequence = torch.from_numpy(c)
        nodes_labels = torch.zeros((sample_num, class_num))
        i = 0
        for la in c:
            nodes_labels[i, la] = 1.0
            i = i + 1
        return nodes_labels

    def _nodes_names_map(self, nodes_names_int):
        """
        我们约定：所有涉及到节点的字典类型，全部是节点的nodes name，str
        所有涉及到节点的矩阵，全部是节点的index，即nodes_names_map[node name]
        note that all dic data with nodes name(str),
        all matrix data with nodes index (int, nodes_names_map[node name])
        """
        nodes_names_map = defaultdict(int)
        i = 0
        for node_name in nodes_names_int:
            nodes_names_map[str(node_name)] = i
            i = i + 1
        return nodes_names_map

    def _hypergraph_cora(self, edges):
        """
        The following hypergraph  is neibhbor-based (1-hop) hypergraph, to generate other kinds of hypergraphs
        please refer the source code of paper:
        Hongxu Chen et al.Multi-level Graph Convolutional Networks for Cross-platform Anchor Link Prediction. KDD2020
        code: https://github.com/sunxiangguo/MGCN

        'Attribute-based'
        # att_col=pickle.load(open('./data/cora/attribute_col_sum.list', 'rb'))  # (超边按照单词的频率降序)
        # print([(i, x) for i, x in enumerate(att_col)])
        H_att = torch.load('./data/cora/attribute_1433.H')[:, 0:1000]
        # print(H_att.shape)
        'Cluster-based'
        H_clu = torch.load('./data/cora/clusters_100.H')
        # print(H_clu.shape)
        'Community-based'
        H_com = torch.load('./data/cora/community_28.H')
        # print(H_com.shape)
        'K-nearest_pro/
        H_knp=torch.load('./data/cora/k_10_nearest_pro.H').float()

        'K-nearest_int'
        H_kni = torch.load('./data/cora/k_10_nearest_int.H').float()
    """

        graph = defaultdict(list)
        for edge in edges:
            graph[edge[0]].append(edge[0])
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[1])
            graph[edge[1]].append(edge[0])
        # 去重复, unique
        for item in graph.items():
            graph[item[0]] = np.unique(item[1])

        indice_matrix = torch.zeros((self.nodes_number, len(graph.keys())))
        # column_names = self.hyperedges.keys()
        col = 0
        for hyperedge in graph.items():
            for node in hyperedge[1]:
                row = self.nodes_names_map[node]
                indice_matrix[row, col] = 1
            col = col + 1

        W_e_diag = torch.ones(indice_matrix.size()[1])

        D_e_diag = torch.sum(indice_matrix, 0)
        D_e_diag = D_e_diag.view((D_e_diag.size()[0]))

        D_v_diag = indice_matrix.mm(W_e_diag.view((W_e_diag.size()[0]), 1))
        D_v_diag = D_v_diag.view((D_v_diag.size()[0]))

        Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
                indice_matrix @ torch.diag(W_e_diag) @ \
                torch.diag(torch.pow(D_e_diag, -1)) @ \
                torch.transpose(indice_matrix, 0, 1) @ \
                torch.diag(torch.pow(D_v_diag, -0.5))

        Theta_inverse = torch.pow(Theta, -1)
        Theta_inverse[Theta_inverse == float("Inf")] = 0

        Theta_I = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
                  indice_matrix @ torch.diag(W_e_diag + torch.ones_like(W_e_diag)) @ \
                  torch.diag(torch.pow(D_e_diag, -1)) @ \
                  torch.transpose(indice_matrix, 0, 1) @ \
                  torch.diag(torch.pow(D_v_diag, -0.5))

        Theta_I[Theta_I != Theta_I] = 0
        Theta_I_inverse = torch.pow(Theta_I, -1)
        Theta_I_inverse[Theta_I_inverse == float("Inf")] = 0

        Laplacian = torch.eye(Theta.size()[0]) - Theta

        fourier_e, fourier_v = torch.symeig(Laplacian, eigenvectors=True)
        # fourier_e, fourier_v = np.linalg.eig(Laplacian)

        wavelets = fourier_v @ torch.diag(torch.exp(-1.0 * fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets_inv = fourier_v @ torch.diag(torch.exp(fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets_t = torch.transpose(wavelets, 0, 1)
        # 根据那篇论文的评审意见，这里用wavelets_t或许要比wavelets_inv效果更好？

        wavelets[wavelets < 0.00001] = 0
        wavelets_inv[wavelets_inv < 0.00001] = 0
        wavelets_t[wavelets_t < 0.00001] = 0

        hypergraph = {"graph": graph,
                      "indice_matrix": indice_matrix,
                      "D_v_diag": D_v_diag,
                      "D_e_diag": D_e_diag,
                      "W_e_diag": W_e_diag,  # hyperedge_weight_flat
                      "laplacian": Laplacian,
                      "fourier_v": fourier_v,
                      "fourier_e": fourier_e,
                      "wavelets": wavelets,
                      "wavelets_inv": wavelets_inv,
                      "wavelets_t": wavelets_t,
                      "Theta": Theta,
                      "Theta_inv": Theta_inverse,
                      "Theta_I": Theta_I,
                      "Theta_I_inv": Theta_I_inverse,
                      }
        return hypergraph

    def _simplegraph_cora(self, edges):
        graph = defaultdict(list)

        for edge in edges:  # 这里的key和value均存的是node_name(str)，不是node_index(按照content的顺序从0开始编号，int)
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        # 去重复，和loop,求D_v
        node_degree_flat = torch.zeros(self.nodes_number)
        A = torch.zeros((self.nodes_number, self.nodes_number))
        for item in graph.items():
            if item[0] in item[1]:
                item[1].remove(item[0])
            graph[item[0]] = np.unique(item[1])
            node_degree_flat[self.nodes_names_map[item[0]]] = len(graph[item[0]])
            for node in graph[item[0]]:
                A[self.nodes_names_map[item[0]], self.nodes_names_map[node]] = 1

        # laplacian
        node_degree_flat_pow = torch.pow(node_degree_flat, -0.5)
        node_degree_flat_pow[node_degree_flat_pow == float("Inf")] = 0  # replace inf with 0
        node_degree_flat_pow[node_degree_flat_pow != node_degree_flat_pow] = 0  # replace nan with 0

        L = torch.eye(self.nodes_number) - torch.diag(node_degree_flat_pow) @ A @ torch.diag(node_degree_flat_pow)
        fourier_e, fourier_v = torch.symeig(L, eigenvectors=True)

        # wavelets
        wavelets = fourier_v @ torch.diag(torch.exp(-1.0 * fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets_inv = fourier_v @ torch.diag(torch.exp(fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets[wavelets < 0.00001] = 0
        wavelets_inv[wavelets_inv < 0.00001] = 0

        # metapath_list
        # metapath_list=[[0,0]]
        # node_type_map
        node_type_list = [0]
        node_type_map = defaultdict(int)  # {'1':0}
        type_node_map = defaultdict(list)  # {0:['1','3','4']}
        for node, _ in graph.items():
            type_node_map[0].append(node)

        for item in graph.items():
            node_type_map[item[0]] = 0

        simple_graph = {"graph": graph,  # {'1':['2','6','10'],}
                        "edges": edges,  # numpy array
                        "node_type_list": node_type_list,
                        # "metapath_list":metapath_list,
                        "node_degree_flat": node_degree_flat,
                        "node_type_map": node_type_map,
                        "type_node_map": type_node_map,
                        "adj": A,
                        "laplacian": L,
                        "fourier_e": fourier_e,
                        "fourier_v": fourier_v,
                        "wavelets": wavelets,
                        "wavelets_inv": wavelets_inv}
        return simple_graph

    def load(self, data_path, data_name, save=False):

        print('start loading...')
        self.data_path = data_path  # "../data/cora/
        content = np.loadtxt(self.data_path + data_name + ".content", dtype=str)
        # print("content\n",content)

        # labels
        nodes_labels_str = content[:, -1]  # str
        self.nodes_labels = self._label_string2matrix(nodes_labels_str)
        self.nodes_number = self.nodes_labels.size()[0]
        # print("self.nodes_number\n",self.nodes_number)

        # node names
        self.nodes_names_int = content[:, 0].astype(np.str)  # .astype(np.int)  # int
        """
        this is a bad variable name, nodes_names_int acturally is str type, not int
        this confusion was caused by some history reasons. please be careful
        """
        print('creating a node map...')
        self.nodes_names_map = self._nodes_names_map(self.nodes_names_int)

        # feature matrix
        nodes_features_int = content[:, 1:-1].astype(np.float)  # int
        print('constructing feature matrix...')
        # sparsity = np.sum(self.nodes_features == 1) * 100.0 / self.nodes_features.size
        self.X_0 = torch.from_numpy(nodes_features_int)
        # print(self.X_0)
        # print(self.X_0.shape)

        # edges
        print('loading edges...')
        self.edges = np.loadtxt(self.data_path + "/" + data_name + ".cites", dtype=str)
        # print(self.edges)
        # print(self.edges.shape)
        """
        default format:
        cora.cites
        src\tdes\ttype
        """
        if data_name in ["cora", 'pubmed', 'aminer', 'spammer']:
            """
            compelete simple graph
            """
            print("construct simple graphs...")
            simple_graph = self._simplegraph_cora(self.edges)
            self.simplegraph_snapshot.append(simple_graph)

            """
            simple graph snapshots, you just need to remove unrelated edges in self.edges, and send
            them into the same function
            simple_graph = self._simplegraph_cora(self.edges)
            self.simplegraph_snapshot.append(simple_graph)
            """

            print("construct hypergraphs...")
            hypergraph = self._hypergraph_cora(self.edges)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            print("load done!")
        elif data_name == "dblp" or data_name == "imdb":
            """
              compelete simple graph
              """
            print("construct simple graphs...")
            simple_graph = self._simplegraph_dblp(self.edges)
            self.simplegraph_snapshot.append(simple_graph)

            print("construct hypergraphs...")

            hypergraph = self._hypergraph_dblp(self.edges)
            self.hypergraph_snapshot.append(hypergraph)
            self.hypergraph_snapshot.append(hypergraph)
            # self.hypergraph_snapshot.append(hypergraph)

            print("load done!")


if __name__ == "__main__":
    data = Data()
    # data.load(data_path='./data/spammer/', data_name='spammer')
    data.load(data_path='./data/cora/', data_name='cora')
