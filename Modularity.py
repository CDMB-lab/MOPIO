import numpy as np
import igraph as ig
from itertools import combinations


def Q(adj_matrix: np.ndarray, label: np.ndarray):
    total = len(label)  # the number of node in the network
    # m = g.ecount()
    m = np.sum(adj_matrix) / 2
    # print("n: {0}, m: {1}".format(total, m))
    q = 0

    for i in range(total):
        for j in range(total):
            if label[i] != label[j]:  # loop continues if node i and node j belong to the different community
                continue
            else:
                q = q + (adj_matrix[i, j] - (len(get_neighbor_list(adj_matrix, i)) * len(get_neighbor_list(adj_matrix, j))) / (2*m))
                # print("i: {}, j: {}, q: {}".format(i, j, q))

    q_result = q / (2*m)
    return q_result


def modularity(g: ig.Graph, label: np.ndarray):
    total = len(label)  # the number of node in the network
    m = g.ecount()
    # print("n: {0}, m: {1}".format(total, m))
    q = 0

    for i in range(total):
        for j in range(total):
            if label[i] != label[j]:  # loop continues if node i and node j belong to the different community
                continue
            else:
                q = q + (g.get_adjacency()[i, j] - (len(g.neighbors(i)) * len(g.neighbors(j))) / (2*m))
                # print("i: {}, j: {}, q: {}".format(i, j, q))

    q_result = q / (2*m)
    return q_result


def q1(g: ig.Graph, label: np.ndarray):
    total = len(label)  # the number of nodes in the network
    # m = int((np.sum(np.sum(g_array, 0)))/2)
    m = g.ecount()
    # print("n: {0}, m: {1}".format(total, m))
    q = 0

    for i in range(total):
        for j in range(total):
            f = 1 if label[i] == label[j] else -1
            tmp = g.get_adjacency()[i, j] - (len(g.neighbors(i)) * len(g.neighbors(j))) / (2*m)
            q = q + tmp * f
            # print("i: {}, j: {}, q: {}".format(i, j, q))

    q_result = q / (4*m)
    return q_result


def q3(g: ig.Graph, label: np.ndarray):
    total = len(label)  # the number of node in the network
    # m = int((np.sum(np.sum(g_array, 0)))/2)
    m = g.ecount()
    # print("n: {0}, m: {1}".format(total, m))
    q = 0
    cluster_num = len(np.unique(label))

    for i in range(cluster_num):
        this_cluster_seq = list(np.where(label == i + 1)[0])
        for j in combinations(this_cluster_seq, 2):
            tmp = g.get_adjacency()[j[0], j[1]] - (len(g.neighbors(j[0])) * len(g.neighbors(j[1]))) / (2*m)
            q = q + tmp

    return q / (2*m)


# 内部度
def q4(g: ig.Graph, label: np.ndarray):
    total = len(label)  # the number of node in the network
    # m = int((np.sum(np.sum(g_array, 0)))/2)
    m = g.ecount()
    # print("n: {0}, m: {1}".format(total, m))
    q = 0
    cluster_num = len(np.unique(label))

    for i in range(cluster_num):
        this_cluster_seq = list(np.where(label == i + 1)[0])
        for j in combinations(this_cluster_seq, 2):
            d1 = len([x for x in g.neighbors(j[0]) if x in this_cluster_seq])
            d2 = len([x for x in g.neighbors(j[1]) if x in this_cluster_seq])
            tmp = g.get_adjacency()[j[0], j[1]] - (d1 * d2) / (2*m)
            q = q + tmp

    return q / (2*m)


def q2(g: ig.Graph, label: np.ndarray):
    total = len(label)  # the number of node in the network
    m = g.ecount()
    # m = int((np.sum(np.sum(g_array, 0))) / 2)
    cluster_num = len(np.unique(label))
    q_result = 0
    for i in range(cluster_num):
        this_cluster_seq = list(np.where(label == i+1)[0])
        _list = [y for x in this_cluster_seq for y in g.neighbors(x) if y in this_cluster_seq]
        print([y for x in this_cluster_seq for y in g.neighbors(x) if y not in this_cluster_seq])
        degree_sum = len(_list)  # the sum of degrees of nodes in group, which group as a independent network
        d = sum([g.degree(x) for x in this_cluster_seq]) # the sum of degrees of nodes inside the group
        print("label: {}, d: {}".format(i+1, d))
        print(_list)
        edge_sum = degree_sum / 2
        tmp = edge_sum/m - (d/(2*m)) ** 2
        print(this_cluster_seq)
        print("degree_sum: {}, tmp: {}".format(degree_sum, tmp))
        q_result = q_result + tmp

    return q_result


def get_neighbor_list(array: np.ndarray, i: int):

    return list(np.where(array[i] > 0)[0])


if __name__ == "__main__":

    # G = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int)
    G = ig.Graph(3, directed=False)
    G.add_edges([(0, 1), (0, 2)])
    L = np.array([2, 1, 2], dtype=int)
    print(modularity(G, L))
    # -0.125
