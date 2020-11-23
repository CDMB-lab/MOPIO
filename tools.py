import numpy as np
from sklearn import metrics
from NMI import nmi
from Modularity import Q
import os


# "real_data/pio/LABEL-moga-moga-football-moga.txt"
def cal_measures_from_label(path: str, m: int, n: int):
    data_arr = txt2np(path)
    data_dict = read_result()
    # file_name = "LABEL-Mar28-011524-karate-20.txt"
    # file_path = path + file_name
    # print(file_path)
    # pre_arr = txt2np(file_path)
    file_name = path.split("/")[-1]
    print(file_name)
    data_name = file_name.split("-")[3]
    print(data_name)
    adj_matrix = txt2np("txt/" + data_name + ".txt")
    true_arr = data_dict[data_name]

    nmi_values_arr = np.zeros(m)
    q_values_arr = np.zeros(m)
    ari_values_arr = np.zeros(m)
    for i in range(m):
        nmi_values_arr[i] = nmi(true_arr, data_arr[i])
        q_values_arr[i] = Q(adj_matrix, data_arr[i])
        ari_values_arr[i] = metrics.adjusted_rand_score(true_arr, data_arr[i])

    nmi_max = np.max(nmi_values_arr)
    nmi_avg = np.mean(nmi_values_arr)
    nmi_std = np.std(nmi_values_arr)
    q_max = np.max(q_values_arr)
    q_avg = np.mean(q_values_arr)
    q_std = np.std(q_values_arr)
    ari_max = np.max(ari_values_arr)
    ari_avg = np.mean(ari_values_arr)
    ari_std = np.std(ari_values_arr)

    _list = file_name.split("-")
    new_name = ""
    for i in range(1, len(_list)-1):
        new_name += _list[i] + "-"
    new_name += _list[-1]
    print(new_name)

    save_result(new_name, nmi_max, nmi_avg, nmi_std, q_max, q_avg, q_std, ari_max, ari_avg,
                ari_std, nmi_values_arr, q_values_arr, ari_values_arr)


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


#  label数组.txt文件生成label的.classes文件
def generate_pairs_classes(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        np_arr[i] = int(txt_arr[i])
    with open(txt_path.split(".")[0] + ".classes", "w") as f1:
        for i in range(np_arr.shape[0]):
            f1.write("{}\t{}\n".format(i+1, np_arr[i]))


def generate_unique_pairs(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        for j in range(txt_arr.shape[1]):
            np_arr[i, j] = int(txt_arr[i, j])
    checked_pairs = []
    pairs = []
    with open(txt_path.split(".")[0] + "_edges.txt", "w") as f1:
        for i in range(np_arr.shape[0]):
            for j in range(np_arr.shape[1]):
                if np_arr[i, j] == 1 and {i+1, j+1} not in checked_pairs:
                    checked_pairs.append({i+1, j+1})
                    pairs.append((i+1, j+1))
        print(pairs)
        for edges in pairs:
            f1.write("{}\t{}\n".format(edges[0], edges[1]))


def community2label(community: list):
    num_v = sum([len(x) for x in community])
    label = np.zeros(num_v, dtype=int)
    for i, v in enumerate(community):
        for each in v:
            label[each] = i+1

    return label


def txt2community(path):
    t = []
    comm = []
    with open(path, "r") as f:
        data = f.readlines()
        for line in data:
            t.append(line.strip("\n"))
        for i in t:
            l1 = i.strip(" ").split(" ")
            l2 = [int(x) for x in l1]
            comm.append(tuple(l2))

    return comm


#  邻接矩阵.txt文件生成边集.txt文件
def generate_pairs(txt_path: str):
    txt_arr = np.loadtxt(txt_path)
    np_arr = np.zeros(txt_arr.shape, dtype=int)
    for i in range(txt_arr.shape[0]):
        for j in range(txt_arr.shape[1]):
            np_arr[i, j] = int(txt_arr[i, j])
    with open(txt_path.split(".")[0] + "_edges.txt", "w") as f1:
        for i in range(np_arr.shape[0]):
            for j in range(np_arr.shape[1]):
                if np_arr[i, j] == 1:
                    f1.write("{}\t{}\n".format(i+1, j+1))


def txt2label(path):
    comm = txt2community(path)
    num_v = sum([len(x) for x in comm])
    label = np.zeros(num_v, dtype=int)
    for i, v in enumerate(comm):
        for each in v:
            label[each-1] = i + 1
    return label


def print_label(path):
    label = txt2label(path)
    with open("maobing.txt", "w") as f:
        for i in label:
            f.write("{}\t".format(i))


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


def cal_ari_from_txt(real_div, path):
    label = txt2label(path)
    return metrics.adjusted_rand_score(real_div, label)


def read_result(path="real_data/RealResult/"):
    dir_list = os.listdir(path)
    data_dict = {}
    for each in dir_list:
        file_path = path + each
        # print(file_path)
        arr = txt2np_1v(file_path)
        data_dict[each.split(".")[0]] = arr

    # print(data_dict)
    return data_dict


# 读取真实数据结果，计算Precision Recall F-measure    # "real_data/pio/"
# "D:/MatlabCode/MOPSO-NET/real_world/"
# "D:/MatlabCode/FN/real_world/"
# "real_data/pio/"
def cal_p_r_f(path="real_data/pio/", file_name="LABEL-moga-moga-football-moga.txt"):

    precision_list = []
    recall_list = []
    f1_score_list = []
    data_dict = read_result()
    # file_name = "LABEL-Mar28-011524-karate-20.txt"
    file_path = path + file_name
    print(file_path)
    pre_arr = txt2np(file_path)
    data_name = file_name.split("-")[3]
    print(data_name)
    true_arr = data_dict[data_name]

    average_mode = "macro"  # 计算模式 micro macro

    for each_arr in pre_arr:
        precision_list.append(metrics.precision_score(true_arr, each_arr, average=average_mode))
        recall_list.append(metrics.recall_score(true_arr, each_arr, average=average_mode))
        f1_score_list.append(metrics.f1_score(true_arr, each_arr, average=average_mode))

    print("{}\n {}\n {}".format(precision_list, recall_list, f1_score_list))
    print("{}\n {}\n {}".format(np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list)))


def trans_label(txt_path: str):
    np_arr = txt2np(txt_path)
    with open("trans_"+txt_path.split("/")[1], "w") as f:
        f.write("index\tmeme\tmoga\tmopso\n")
        for j in range(np_arr.shape[1]):
            f.write("{}\t{}\t{}\t{}\n".format(j+1, np_arr[0][j], np_arr[1][j], np_arr[2][j]))


def trans_fast_label(txt_path: str):
    np_arr = txt2np_1v(txt_path)
    with open("trans_"+txt_path.split("/")[1], "w") as f:
        f.write("index\tfast\n")
        for j in range(len(np_arr)):
            f.write("{}\t{}\n".format(j+1, np_arr[j]))

if __name__ == "__main__":
    cal_p_r_f()
    # read_result()
