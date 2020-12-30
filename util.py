import time
import pio1
import scipy.io as sio
from matplotlib import pylab as plt
import numpy as np
import igraph as ig
from Modularity import modularity, Q
from NMI import nmi
from sklearn import metrics
import meme_main


def generate_pairs_classes(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        np_arr[i] = int(txt_arr[i])
    with open(txt_path.split(".")[0] + ".classes", "w") as f1:
        for i in range(np_arr.shape[0]):
            f1.write("{}\t{}\n".format(i+1, np_arr[i]))


def generate_pairs(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        for j in range(txt_arr.shape[1]):
            np_arr[i, j] = int(txt_arr[i, j])
    with open(txt_path.split(".")[0] + ".pairs", "w") as f1:
        for i in range(np_arr.shape[0]):
            for j in range(np_arr.shape[1]):
                if np_arr[i, j] == 1:
                    f1.write("{}\t{}\n".format(i+1, j+1))


def txt2gml(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        for j in range(txt_arr.shape[1]):
            np_arr[i, j] = int(txt_arr[i, j])

    edges_list = []
    for i in range(np_arr.shape[0]):
        for j in range(i+1, np_arr.shape[1]):
            if np_arr[i, j] == 1:
                edges_list.append((i, j))

    generate_gml_file(edges_list, txt_path.split(".")[0] + ".gml")


def txt2np(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        for j in range(txt_arr.shape[1]):
            np_arr[i, j] = int(txt_arr[i, j])

    return np_arr


def txt2np_1v(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(len(txt_arr), dtype=int)
    for i in range(len(txt_arr)):
        np_arr[i] = int(txt_arr[i])

    return np_arr


def generate_gml_file(edges, filename):  # edges: [(0, 1), (0, 2), (1, 2), ...]
    """
    generate gml file
    """
    vertex = np.unique(edges)
    fp_out = open(filename, 'w')
    fp_out.write("graph\n")
    fp_out.write("[\n")
    for node in vertex:
        fp_out.write("\tnode\n")
        fp_out.write("\t[\n")
        fp_out.write("\t\tid " + str(node) + "\n")
        # insert other node attributes here
        fp_out.write("\t]\n")

    for e in edges:
        fp_out.write("\tedge\n")
        fp_out.write("\t[\n")
        fp_out.write("\t\tsource " + str(e[0]) + "\n")
        fp_out.write("\t\ttarget " + str(e[1]) + "\n")
        # insert other edge attributes here
        fp_out.write("\t]\n")

    fp_out.write("]")
    fp_out.close()


'''
graph
[
    node
    [
        id 1
    ]
    node
    [
        id 2
    ]
    edge
    [
        source 2
        target 1
    ]
]
'''

data = pio1.read_result("data.pkl")
karate = data["karate"]
football = data["football"]
dolphins = data["dolphins"]
fb50 = data["fb50"]
polbooks = data["polbooks"]

real_data = {0: "karate", 1: "football", 2: "fb50", 3: "polbooks", 4: "dolphins"}
real_label = {0: karate, 1: football, 2: fb50, 3: polbooks, 4: dolphins}

data_dict = {0: real_data}
label_dict = {0: real_label}
p_and_g = {0: (50, 50), 1: (50, 100), 2: (100, 100), 3: (100, 150), 4: (150, 150)}


# file_path = "gml/"
def pio_run():
    save_path = "real_data/pio/"
    n = 10
    data_type = 0
    for data_index in range(3, 4):
        # data_index = 0
        data_path = "gml/" + data_dict[data_type][data_index] + ".gml"
        for para_index in range(0, 1):

            pop_size = p_and_g[para_index][0]
            iteration_num = p_and_g[para_index][1]

            leader_arr = np.zeros(n, dtype=int)
            max_nmi_values_arr = np.zeros(n)
            nmi_values_arr = np.zeros(n)
            q_values_arr = np.zeros(n)
            ari_values_arr = np.zeros(n)
            g = ig.Graph.Read_GML(data_path)
            label = np.empty((n, g.get_adjacency().shape[0]), dtype=int)
            for i in range(n):
                # imp.reload(pio1)
                # pio1.path = file_path + data_dict[data_index] + ".gml"
                pio1.real_label = label_dict[data_type][data_index]
                p, p_div, a_div, leader = pio1.pio_main(pop_size, iteration_num, data_path)
                leader_arr[i] = leader
                label[i] = a_div[leader]
                nmi_values_arr[i] = nmi(label_dict[data_type][data_index], a_div[leader])
                q_values_arr[i] = modularity(pio1.graph, a_div[leader])
                ari_values_arr[i] = metrics.adjusted_rand_score(label_dict[data_type][data_index], a_div[leader])
                print("####  {}  ####".format(i + 1))

            nmi_max = np.max(nmi_values_arr)
            nmi_avg = np.mean(nmi_values_arr)
            nmi_std = np.std(nmi_values_arr)
            q_max = np.max(q_values_arr)
            q_avg = np.mean(q_values_arr)
            q_std = np.std(q_values_arr)
            ari_max = np.max(ari_values_arr)
            ari_avg = np.mean(ari_values_arr)
            ari_std = np.std(ari_values_arr)
            # _sum2 = _sum2 / n
            localtime = time.asctime(time.localtime(time.time()))
            time_list = localtime.split(" ")
            if "" in time_list:
                time_list.remove("")
            file_name = time_list[1] + time_list[2] + "-" + time_list[3].replace(":", "") + "-" + data_dict[data_type][
                data_index] + "-" + str(n) + ".txt"
            save_divs(save_path, file_name, label)
            save_result(save_path + file_name, nmi_max, nmi_avg, nmi_std, q_max, q_avg, q_std, ari_max, ari_avg,
                        ari_std,
                        nmi_values_arr, q_values_arr, ari_values_arr)

    return "mission complete"


# file_path = "txt/"
# data_dict = {0: "karate", 1: "football", 2: "fb50", 3: "polbooks", 4: "dolphins"}
def meme_run():
    save_path = "real_data/meme/"
    n = 20
    data_type = 0
    for data_index in range(1, 2):
        # data_index = 1
        data_path = "txt/" + data_dict[data_type][data_index] + ".txt"
        for para_index in range(1, 2):
            pop_size = p_and_g[para_index][0]
            iteration_num = p_and_g[para_index][1]

            print(data_path)

            nmi_values_arr = np.zeros(n)
            q_values_arr = np.zeros(n)
            ari_values_arr = np.zeros(n)
            A = txt2np(data_path)
            label = np.empty((n, A.shape[0]), dtype=int)
            for i in range(n):

                label[i] = meme_main.main(pop_size, iteration_num, data_path)
                # print(label[i])
                nmi_values_arr[i] = nmi(label_dict[data_type][data_index], label[i])
                q_values_arr[i] = Q(A, label[i])
                ari_values_arr[i] = metrics.adjusted_rand_score(label_dict[data_index], label[i])
                print("####  {}  ####".format(i + 1))

            nmi_max = np.max(nmi_values_arr)
            nmi_avg = np.mean(nmi_values_arr)
            nmi_std = np.std(nmi_values_arr)
            q_max = np.max(q_values_arr)
            q_avg = np.mean(q_values_arr)
            q_std = np.std(q_values_arr)
            ari_max = np.max(ari_values_arr)
            ari_avg = np.mean(ari_values_arr)
            ari_std = np.std(ari_values_arr)
            # _sum2 = _sum2 / n
            localtime = time.asctime(time.localtime(time.time()))
            time_list = localtime.split(" ")
            if "" in time_list:
                time_list.remove("")
            file_name = time_list[1] + time_list[2] + "-" + time_list[3].replace(":", "") + "-" + data_dict[data_type][
                data_index] + "-" + str(n) + ".txt"
            save_divs(save_path, file_name, label)
            save_result(save_path+file_name, nmi_max, nmi_avg, nmi_std, q_max, q_avg, q_std, ari_max, ari_avg, ari_std,
                                                      nmi_values_arr, q_values_arr, ari_values_arr)

    return "mission complete"


def save_divs(save_path, file_name, label):
    path = save_path + "LABEL-" + file_name
    n = label.shape[0]
    m = label.shape[1]
    with open(path, "w") as f:
        for i in range(n):
            for j in range(m):
                f.write("{}\t".format(label[i][j]))

            f.write("\n")

def save_result(path, nmi_max, nmi_avg, nmi_std, q_max, q_avg, q_std, ari_max, ari_avg, ari_std,
                                                      nmi_values_arr, q_values_arr, ari_values_arr):
    with open(path, "w") as f:
        f.write("nmi_max: {}\t"
                "nmi_avg: {}\t"
                "nmi_std: {}\n"
                "q_max: {}\t"
                "q_avg: {}\t"
                "q_std: {}\n"
                "ari_max: {}\t"
                "ari_avg: {}\t"
                "ari_std: {}\n"
                "nmi_values_arr: {}\n"
                "q_values_arr: {}\n"
                "ari_values_arr: {}\n".format(nmi_max, nmi_avg, nmi_std, q_max, q_avg, q_std, ari_max, ari_avg, ari_std,
                                              nmi_values_arr, q_values_arr, ari_values_arr))


if __name__ == "__main__":
    # run1()
    pio_run()
    # for i in range(1):
    #     pio_run()
    #     print("################################################# {} ###############################################".format(i))
