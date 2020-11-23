import igraph as ig
import numpy as np
import copy
# import dfs
import random
import math
import pickle
import time
import itertools
from NMI import nmi


def get_x_center_index(label: np.ndarray) -> int:
    nra_values = [abs(nra(x)) for x in label]  # 正值
    nra_sum = sum(nra_values)
    t = random.random() * nra_sum
    for key, value in enumerate(nra_values):
        t -= value
        if t < 0:
            return key

    return len(label) - 1


def sorted_by_crowding_distance(function_values: list, index: list):
    global d_values_sorted
    function_value = [function_values[x] for x in index]  # [(0.23, 0.123), (0.545, 0.3463), ...]
    num = len(function_value)
    if num == 1 or num == 2:
        return list(range(num))
    f_num = len(function_value[0])
    distance = np.zeros(num)
    for f in range(f_num):
        # f_values = np.array([(x, x[f]) for x in function_value])

        f_values = [(x, function_value[x][f]) for x in range(num)]
        # f_values: [(1, 3), (2, 1), (3, 6), (4, 0), (5, 4)]
        f_values_sorted = sorted(f_values, key=lambda x: x[1], reverse=False)
        # f_values_sorted: [(4, 0), (2, 1), (1, 3), (5, 4), (3, 6)]
        _max = f_values_sorted[-1][1] - f_values_sorted[0][1]
        if _max == 0:
            # print("_max: {}, num: {}".format(_max, num))
            _max = f_values_sorted[-1][1]  # 防止分母为零
        _list = [(f_values_sorted[x+1][1] - f_values_sorted[x-1][1])/_max for x in range(1, num-1)]
        _list.insert(0, inf)
        _list.append(inf)
        # print(_list)
        # distance += np.array(_list)
        for i in range(num):
            distance[f_values_sorted[i][0]] += _list[i]

    d_values = [(x, distance[x]) for x in range(num)]
    d_values_sorted = sorted(d_values, key=lambda x: x[1], reverse=True)  # 按距离递减排序
    # print(len(d_values_sorted))
    # print(d_values_sorted)
    # print(d_values_sorted)

    return [x[0] for x in d_values_sorted]


def get_archive_index(div: np.ndarray):
    num = len(div)
    c = []
    for i in range(num):
        c += ((i+1)*[num-1-i])
    return random.choice(c)


def pio_main(population_size: int, iteration_num: int, data_path: str):
    s = time.time()
    global population_num
    global archive_num
    global pigeons_function_value
    global best_pigeons_function_value
    global archive_function_value
    global best_pigeons
    global best_pigeons_div
    global archive_div  # 存储全局最优解
    global archive_array
    global graph, real_label, graph_array, neighbors_list, path

    path = data_path
    graph = get_graph(path)
    real_label = np.zeros(shape[0])
    archive_array = np.zeros(shape[1], dtype=int)
    archive_div = np.zeros(shape[1], dtype=int)
    graph_array = matrix2array(graph.get_adjacency())
    neighbors_list = get_neighbor_list(graph_array)

    best_pigeons = np.zeros(shape[1], dtype=int)
    best_pigeons_div = np.zeros(shape[1], dtype=int)

    leader = np.zeros(1, dtype=int)

    population_num = population_size
    pigeons = get_main_variable(population_size, data_path)  # 初始化种群，初始化个体最优变量
    best_pigeons = pigeons

    # 初始化全局最优
    p_div = get_division_scheme(pigeons)
    best_pigeons_div = p_div
    pigeons_function_value = [get_nra_and_rc_value(x) for x in p_div]
    best_pigeons_function_value = [x for x in pigeons_function_value]
    # _dict = assign_rank(0)  # 0: pigeons_function_value; 1: archive_function_value
    _dict = ranked(pigeons_function_value)
    for index in _dict[1]:
        archive_div = np.row_stack((archive_div, p_div[index]))
        archive_array = np.row_stack((archive_array, pigeons[index]))
        archive_num += 1
    archive_div = archive_div[1:]
    archive_array = archive_div[1:]

    for nc in range(1, iteration_num):
        kids = pigeons
        s1 = time.time()
        for each in range(population_size):

            sample_num = round(shape[1] * 0.8)  # 遗传率
            rand_seq = random.sample(range(shape[1]), sample_num)  # 继承序列
            best_pigeons_weight = 1 - math.log(nc + 1, iteration_num)
            if best_pigeons_weight < 0.2:
                best_pigeons_weight = 0.2
            # center_weight = 0.5
            best_pigeons_num = round(sample_num * best_pigeons_weight)
            best_pigeons_seq = random.sample(rand_seq, best_pigeons_num)
            global_best_seq = [x for x in rand_seq if x not in best_pigeons_seq]
            # tmp_pigeons = pigeons[each]  # 0: pigeons 1: best_pigeons
            # global_best = random.choice(archive_div)  # 重新选择global best

            global_best = get_archive_index(archive_div)  # 全局最优解索引 archive_array[0]
            g_pigeons = archive_div[global_best]

            kids[each][best_pigeons_seq] = [x for x in best_pigeons[each][best_pigeons_seq]]
            # kids[each][global_best_seq] = [x for x in g_pigeons[global_best_seq]]
            kids[each][global_best_seq] = get_seq(g_pigeons, global_best_seq)

            mutation_num = round(shape[1] * P)
            mutation_seq = random.sample(range(shape[1]), mutation_num)  # 在整个序列突变
            # pigeons[each][mutation_seq] = [random.choice(neighbors_list[x]) for x in mutation_seq]
            kids[each][mutation_seq] = [choose_by_cc(neighbors_list[x]) for x in mutation_seq]

            # tmp_pigeon = pigeons[each]
            # tmp_pigeon.shape = (1, shape[1])
            # pigeons_function_value[each] = get_nra_and_rc_value(get_division_scheme(tmp_pigeon)[0])
            # best_pigeons[each] = pigeons[each] if check(pigeons_function_value, best_pigeons_function_value, each, each) != -1 else best_pigeons[each]  # 子代不是差的就接受

        e1 = time.time()
        print("子代生成时间: {}".format(e1-s1))

        s2 = time.time()
        # 合并子代与亲代，分层
        pigeons = np.vstack((pigeons, kids))
        p_div = get_division_scheme(pigeons)
        pigeons_function_value = [get_nra_and_rc_value(x) for x in p_div]
        _dict = ranked(pigeons_function_value)

        # f1装载进archive
        for index in _dict[1]:
            archive_div = np.row_stack((archive_div, p_div[index]))
            archive_array = np.row_stack((archive_array, pigeons[index]))
            archive_num += 1

        # 对archive非支配分层，保留archive.f1,即archive = archive.f1
        archive_div = np.unique(archive_div, axis=0)
        archive_function_value = [get_nra_and_rc_value(x) for x in archive_div]
        archive_num = len(archive_div)
        print("ranked前: {}".format(archive_num))
        archive_dict = ranked(archive_function_value)
        # print(archive_dict)
        tmp_array = np.zeros(shape[1], dtype=int)
        tmp_div = np.zeros(shape[1], dtype=int)
        for each_index in archive_dict[1]:
            tmp_array = np.row_stack((tmp_array, archive_array[each_index]))
            tmp_div = np.row_stack((tmp_div, archive_div[each_index]))

        archive_array = tmp_array[1:]
        archive_div = tmp_div[1:]
        archive_num = len(archive_div)
        e2 = time.time()
        print("子代与亲代分层, Archive分层时间: {} , 分层保留: {}, Archive: {}".format(e2-s2, len(_dict[1]), archive_num))

        s3 = time.time()
        # 保留前N个体
        # total_num = 2 * population_size
        level_num = len(_dict)
        # print(_dict)
        pop_num = population_size
        while pop_num - len(_dict[level_num]) >= 0:
            # print("level_num: {}".format(level_num))
            # total_num -= len(_dict[level_num])
            pop_num -= len(_dict.popitem()[1])
            level_num -= 1
        # print(_dict[level_num])
        # print("pop_num: {}".format(pop_num))
        while pop_num > 0:
            _dict[level_num].pop()
            pop_num -= 1

        remained_seq = []
        while level_num > 0:
            remained_seq += _dict[level_num]
            level_num -= 1
        # print("_dict: {}".format(len(remained_seq)))

        # 更新个体最优
        pigeons = pigeons[remained_seq]
        # best_pigeons = pigeons
        p_div = get_division_scheme(pigeons)
        pigeons_function_value = [get_nra_and_rc_value(x) for x in p_div]
        for i in range(population_size):
            best_pigeons[i] = pigeons[i] if check(pigeons_function_value, best_pigeons_function_value, i, i) != -1 else best_pigeons[i]
        # print("pass!")
        e3 = time.time()
        print("保留N, 更新个体最优时间: {}, 当前迭代次数: {}".format(e3-s3, nc+1))
        # print("iteration_num: {0}".format(nc + 1, population_size))
        if nc + 1 == iteration_num:
            archive_function_value = [get_nra_and_rc_value(x) for x in archive_div]
            leader[0] = get_leader()  # get an index of leader from archive
            save_result(pigeons, p_div, archive_div, population_size, iteration_num, leader[0])
            print("循环结束")
        # print("Nc({}) after updating, pigeons: ".format(nc))
        # print(pigeons)
        # print("Nc({})after updating, velocities: ".format(nc))
        # print(velocities)

    # print("pigeons: ")
    # print(pigeons)
    # print("velocities: ")
    # print(velocities)
    e = time.time()
    print("方法总时长: {}".format(e-s))
    return pigeons, p_div, archive_div, leader[0]


def get_graph(gml_dir="data/karate.gml") -> ig.Graph:
    g = ig.Graph.Read_GML(gml_dir)
    global FN
    FN = gml_dir
    global shape
    shape = g.get_adjacency().shape
    return g


def get_seq(x_best_array: np.ndarray, best_seq_list: list) -> list:
    result = []
    for x in best_seq_list:
        comm = list(np.where(x_best_array == x_best_array[x])[0])
        comm.remove(x)
        neighbors = graph.neighbors(x)
        intersection = [x for x in comm if x in neighbors]
        if len(intersection) == 0:
            print("### Error: a list is empty. ###")
        result.append(random.choice(intersection))
    return result


def choose_by_degree(neighbor_seq: list):
    # weight_seq = [get_clustering_coefficient(x) for x in neighbor_seq]
    weight_seq = [graph.degree(x) for x in neighbor_seq]
    # weight_sum = sum(weight_seq)
    value_list = []
    for key, value in enumerate(weight_seq):
        value_list += [neighbor_seq[key]] * value
    # t = random.random() * weight_sum
    # for key, value in enumerate(weight_seq):
    #     t -= value
    #     if t < 0:
    #         break
    # return neighbor_seq[key]
    return random.choice(value_list)


def choose_by_cc(neighbor_seq: list):
    weight_seq = [get_clustering_coefficient(x) for x in neighbor_seq]
    # weight_seq = [graph.degree(x) for x in neighbor_seq]
    weight_sum = sum(weight_seq)
    # value_list = []
    # for key, value in enumerate(weight_seq):
    #     value_list += [neighbor_seq[key]] * value
    t = random.random() * weight_sum
    for key, value in enumerate(weight_seq):
        t -= value
        if t < 0:
            return neighbor_seq[key]

    return neighbor_seq[-1]
    # return random.choice(value_list)


def get_clustering_coefficient(v: int):
    neighbors = graph.neighbors(v)
    neighbors_nums = len(neighbors)
    if neighbors_nums == 1:
        return 1
    neighbors_tuple = itertools.combinations(neighbors, 2)
    edges_sum = 0
    for i in neighbors_tuple:
        if graph.are_connected(i[0], i[1]):
            edges_sum += 1
    return (2 * edges_sum) / (neighbors_nums * (neighbors_nums-1))


def save_result(p, p_div, a_div, p_size, iter_num, l: int):
    data_dict = {"pigeons": p, "p_div": p_div, "archive_div": a_div, "leader": l}
    localtime = time.asctime(time.localtime(time.time()))
    time_list = localtime.split(" ")
    if "" in time_list:
        time_list.remove("")
    remarks = ""
    save_path = "tmp/"
    file_name = path + time_list[1] + time_list[2] + "-" + time_list[3].replace(":", "") + \
        "-" + FN.split("/")[1].split(".")[0] + "-" + str(p_size) + "-" + str(iter_num)
    if len(remarks) > 0:
        file_name = file_name + "-" + remarks
    file_name = file_name + ".pkl"
    # file_name = ?
    with open(file_name, "wb") as f:
        pickle.dump(data_dict, f)


def read_result(filename):
    _dir = ""
    filename = _dir + filename
    with open(filename, "rb") as f:
        result_dict = pickle.load(f)

    return result_dict


def get_offset(location: np.ndarray, velocity: np.ndarray):
    loc = np.zeros(shape[1], dtype=int)
    for i in range(shape[1]):
        loc[i] = neighbors_list[i][(neighbors_list[i].index(location[i]) + velocity[i]) % len(neighbors_list[i])]

    return loc


def get_main_variable(population_size: int, data_path: str):
    # global pigeons_function_value
    g = get_graph(data_path)
    g_array = matrix2array(g.get_adjacency())
    neighbors = get_neighbor_list(g_array)
    pigeons = init_population(population_size, neighbors)
    # pigeons_function_value = [get_nra_and_rc_value(x) for x in pigeons]
    # velocities = init_velocity(population_size)
    # p_div = get_division_scheme(pigeons)
    return pigeons
    # g, g_array, neighbors, pigeons, velocities, p_div= pio.get_main_variable()


def get_variable(population_size: int):
    # global pigeons_function_value
    g = get_graph()
    g_array = matrix2array(g.get_adjacency())
    neighbors = get_neighbor_list(g_array)
    pigeons = init_population(population_size, neighbors)
    # pigeons_function_value = [get_nra_and_rc_value(x) for x in pigeons]
    p_div = get_division_scheme(pigeons)
    return g, g_array, neighbors, pigeons, p_div
    # g, g_array, neighbors, pigeons, velocities, p_div= pio.get_main_variable()


def get_r():
    com = [x for x in itertools.combinations(range(len(archive_function_value)), 2)]
    min1 = abs(archive_function_value[com[0][0]][0] - archive_function_value[com[0][1]][0])
    # min1 = archive_function_value[0][0]
    # max2 = archive_function_value[0][1]
    # for i in range(1, len(archive_function_value)):
    #     if archive_function_value[i][0] < min1:
    #         min1 = archive_function_value[i][0]
    #     if archive_function_value[i][1] > max2:
    #         max2 = archive_function_value[i][1]
    for each_tuple in com:
        v = abs(archive_function_value[each_tuple[0]][0] - archive_function_value[each_tuple[1]][0])
        if v < min1 and v != 0:
            min1 = v

    # r = ((min1**2 + max2**2)**0.5) * 0.125 * 0.09
    r = min1 * 1.2
    print("r: {}".format(r))
    return r


def get_leader() -> int:
    a_num = len(archive_div)
    rank_score = np.array(range(a_num))
    rank_score[1] = rank_score[0]
    q_arr = [graph.modularity(x) for x in archive_div]
    q_arr = [x/max(q_arr) for x in q_arr]
    q_dict = {}
    for i in range(a_num):
        q_dict[i] = q_arr[i]

    q_sorted_list = sorted(q_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(a_num):
        index = q_sorted_list[i][0]
        rank_score[index] += i

    print(q_sorted_list)
    print(rank_score)
    rank_score_list = [x for x in rank_score]
    # r_num = len(rank_score_list)
    a_sorted_list = sorted_by_crowding_distance(archive_function_value, list(range(len(archive_function_value))))
    print("d_values_sorted:")
    print(d_values_sorted)
    result_list = []
    # for i in range(a_num):
    #     # min_score = min(rank_score_list)
    #     # min_score_index = np.where(rank_score == rank_score_list[i])[0][0]
    #     for each in d_values_sorted:
    #
    #         if each[0] == i and each[1] > elim_threshold:
    #             result_list.append((min_score_index, d_values_sorted[min_score_index]))
    #             break
    # print(result_list)
    for each in d_values_sorted:
        if each[1] > elim_threshold:
            result_list.append(each)
    print("result_list:")
    print(result_list)
    del rank_score_list[len(result_list):]
    print("rank_score_list: {}".format(rank_score_list))

    # 把 rank_score_list 绑定索引存入列表
    rs_list = []
    for i in range(len(result_list)):
        rs_list.append((i, rank_score_list[i]))
    rs_sorted = sorted(rs_list, key=lambda x: x[1], reverse=False)
    print("rs_sorted: {}".format(rs_sorted))

    ans = [rs_sorted[0][0]]  # 把rs_sorted第一个个体值存入
    # 选top个体，差值不超过d_value
    for i in range(1, len(result_list)):
        print("{} - {}".format(result_list[rs_sorted[i][0]][1], result_list[rs_sorted[i-1][0]][1]))
        if abs(result_list[rs_sorted[i][0]][1] - result_list[rs_sorted[i-1][0]][1]) < dist_d_value:
            ans.append(rs_sorted[i][0])
        else:
            break
    print("ans: {}".format(ans))
    if result_list[ans[0]][1] == inf * f_num:
        return ans[0]
    else:
        return random.choice(ans)
    # selected_index = np.where(rank_score == min(rank_score))[0][0]
    # for each in result_list:
    #     if each[0] == selected_index:
    #         if each[1] == inf * f_num:
    #             return selected_index
    #         break
    #
    # return np.where(rank_score == min(rank_score))[0][0]


def matrix2array(g_matrix: ig.datatypes.Matrix):
    array = np.zeros(g_matrix.shape, dtype=int)
    for i in range(g_matrix.shape[0]):
        for j in range(g_matrix.shape[1]):
            array[i, j] = g_matrix[i, j]

    return array


def get_neighbor_list(array: np.ndarray):
    neighbor_list = []
    for i in range(array.shape[0]):
        neighbor_list.append(list(np.where(array[i] > 0)[0]))

    return neighbor_list


def create_graph_dict(parent, v_count):
    parent_list = list()
    individual_dict = dict()
    for each_parent in range(len(parent)):
        for i in range(v_count):
            individual_dict[i] = list([])
        for each_loci in range(v_count):
            v = parent[each_parent][each_loci]
            if v not in individual_dict[each_loci]:
                individual_dict[each_loci].append(v)
            if each_loci not in individual_dict[v]:
                individual_dict[v].append(each_loci)

        parent_list.append(copy.deepcopy(individual_dict))
    return parent_list


def get_division_scheme(population: np.ndarray):
    # parent_label = list()
    # visited = set()
    labels = np.zeros(population.shape, dtype=int)  # 存储个体解方案的分区情况
    parent_dict_list = create_graph_dict(population, population.shape[1])
    for i in range(len(population)):
        visited = set()  # 遍历每个连通分量时的临时visited数组
        v = set()  # 个体i本次遍历的全局visited数组
        c_num = 1  # 1号分区
        for j in range(population.shape[1]):
            if j not in visited and j not in v:
                visited = set()
                dfs(parent_dict_list[i], j, visited)
                v = v.union(visited)
                # print(visited)
                # print("division_num:" + str(c_num))
                for each in visited:
                    labels[i, each] = c_num
                c_num += 1

    return labels


def kkm(label: np.ndarray):
    n = len(label)
    m = len(set(label))
    inside_links = np.zeros((2, m), dtype=int)
    for i in range(m):
        com = np.where(label == i + 1)
        com = list(com[0])
        inside_links[1][i] = len(com)
        for j in com:
            inside_links[0][i] += sum([1 if x in com else 0 for x in list(np.where(graph_array[j] > 0)[0])])

    inside_links[0] = inside_links[0] / 2
    # print("inside_links:\n" + str(inside_links))
    kkm_value = 2 * (n - m) - sum(inside_links[0] / inside_links[1])
    return kkm_value


def nra(label: np.ndarray):
    m = len(set(label))
    inside_links = np.zeros((2, m), dtype=int)
    for i in range(m):
        com = np.where(label == i + 1)
        com = list(com[0])
        inside_links[1][i] = len(com)
        for j in com:
            inside_links[0][i] += sum([1 if x in com else 0 for x in list(np.where(graph_array[j] > 0)[0])])

    inside_links[0] = inside_links[0] / 2
    # print("inside_links:\n" + str(inside_links))
    nra_value = - sum(inside_links[0] / inside_links[1])
    return nra_value


def rc(label: np.ndarray):
    n = len(label)
    m = len(set(label))
    among_links = np.zeros((2, m), dtype=int)
    for i in range(m):
        com = np.where(label == i + 1)
        com = list(com[0])
        among_links[1][i] = len(com)
        for j in set(range(n)).difference(com):
            among_links[0][i] += sum([1 if x in com else 0 for x in list(np.where(graph_array[j] > 0)[0])])

    # print("among_links:\n" + str(among_links))
    rc_value = sum(among_links[0] / among_links[1])
    return rc_value


def get_kkm_and_rc_value(label: np.ndarray):
    return tuple((kkm(label), rc(label)))


def get_nra_and_rc_value(label: np.ndarray):
    return tuple((nra(label), rc(label)))


def check_dominance(label: int, l1: int, l2: int):  # 0: p, p 1: a, a 2: p, bp
    function_value1 = function_value2 = pigeons_function_value
    if label == 1:
        function_value1 = function_value2 = archive_function_value
    elif label == 2:
        function_value1 = pigeons_function_value
        function_value2 = best_pigeons_function_value
    flag = 0
    # ans1 = [kkm(s1, l1), rc(s1, l1)]  # minimize KKM and RC
    # ans2 = [kkm(s2, l2), rc(s2, l2)]
    ans1 = function_value1[l1]  # minimize NRA and RC
    ans2 = function_value2[l2]
    if (ans1[0] < ans2[0] and 0 < ans1[1] <= ans2[1]) or (ans1[0] <= ans2[0] and 0 < ans1[1] < ans2[1]):
        flag = 1
    elif (ans1[0] > ans2[0] and ans1[1] >= ans2[1] > 0) or (ans1[0] >= ans2[0] and ans1[1] > ans2[1] > 0):
        flag = -1
    # print(ans1, ans2)
    # print(flag)

    return flag


def ranked(function_value: list) -> dict:
    num = len(function_value)
    now_seq = list(range(num))
    ranked_dict = {}
    i = 1
    while len(now_seq) > 0:
        seq1, seq2 = ranking(function_value, now_seq)
        # print("seq1: {}, seq2: {}".format(seq1, seq2))
        if len(seq1) == 0:
            return ranked_dict
        # print("seq1: {}, seq2: {}".format(seq1, seq2))
        order_seq = sorted_by_crowding_distance(function_value, seq1)
        ranked_dict[i] = [seq1[x] for x in order_seq]
        i += 1
        now_seq = [x for x in seq2]
    return ranked_dict


def check(function_value1, function_value2, i, j):
    flag = 0
    ans1 = function_value1[i]  # minimize NRA and RC
    ans2 = function_value2[j]
    if (ans1[0] < ans2[0] and 0 < ans1[1] <= ans2[1]) or (ans1[0] <= ans2[0] and 0 < ans1[1] < ans2[1]):
        flag = 1
    elif (ans1[0] > ans2[0] and ans1[1] >= ans2[1] > 0) or (ans1[0] >= ans2[0] and ans1[1] > ans2[1] > 0):
        flag = -1
    return flag


def ranking(function_value, seq):
    num = len(function_value)
    dominated_index = np.zeros(num, dtype=int)
    dominated_num = np.zeros(num, dtype=int)  # 个体被支配解的数量
    dominating_seq = []
    # _seq = list(range(len(seq)))
    for i in seq:
        if dominated_index[i] == 1:
            continue
        for j in seq:
            if i == j:
                continue
            if check(function_value, function_value, i, j) == -1:
                dominated_num[i] += 1
            elif check(function_value, function_value, i, j) == 1:
                dominated_index[j] = 1

        if dominated_num[i] == 0:
            dominating_seq.append(i)

    return dominating_seq, [x for x in seq if x not in dominating_seq]


def assign_rank(label: int):
    num = population_num if label == 0 else archive_num
    rank = np.zeros(num, dtype=int)  # 个体排名
    floor_index_dict = {}  # 分层集合
    dominating_dict = {}  # 支配解的集合
    dominated_index = np.zeros(num, dtype=int)
    dominated_num = np.zeros(num, dtype=int)  # 个体被支配解的数量
    for p_index in range(num):
        if dominated_index[p_index] == 1:
            continue
        for q_index in range(num):
            if p_index == q_index:
                continue
            if check_dominance(label, p_index, q_index) == -1:
                dominated_num[p_index] += 1
            elif check_dominance(label, p_index, q_index) == 1:
                dominated_index[q_index] = 1
                add_num_into_dict(p_index, q_index, dominating_dict)
            # elif check_dominance(p, labels[p_index], q, labels[q_index]) == -1:
            #     dominated_num[p_index] += 1

        # if dominated_num[p_index] == 0:
        if dominated_num[p_index] == 0:
            rank[p_index] = 1
            add_num_into_dict(1, p_index, floor_index_dict)
    # '''
    i = 1
    # loop_num = 0
    # print("floor_index_dict:" + str(floor_index_dict))
    # print("dominating_dict:" + str(dominating_dict))
    while len(floor_index_dict[i]) > 0:
        tmp_list = list()
        for p_index in floor_index_dict[i]:
            # print("floor_index_dict[{0}]: {1}".format(i, floor_index_dict[i]))
            # print("p_index:{0}, dominated_num:{1}".format(p_index, dominated_num))
            try:
                for q_index in dominating_dict[p_index]:
                    dominated_num[q_index] -= 1
                    if dominated_num[q_index] == 0:
                        rank[q_index] = i + 1
                        tmp_list.append(q_index)  # store index or value ?

            except KeyError:
                pass
                # print("KeyError: " + str(p_index))

        i += 1
        add_list_into_dict(i, tmp_list, floor_index_dict)
    # '''
    return floor_index_dict


def assign_crowding_distance(floor_dict: dict, index_in_floor: int, population: np.ndarray, labels: np.ndarray):
    solutions_in_floor = floor_dict[index_in_floor]  # tmp_floor为引用变量
    num = len(solutions_in_floor)
    solutions_distance = np.zeros((2, num))
    avg_distance = np.zeros(num)
    solution_indices = list()
    avg_distance_and_indices_dict = {}
    for i in range(2):
        # print("before sorting:" + str(floor_dict))
        sorted_indices_list, sorted_values_list = sort_floor(solutions_in_floor, i, population, labels)
        solution_indices.append(sorted_indices_list)
        # print("after sorting:" + str(sorted_list))
        solutions_distance[i][0] = solutions_distance[i][num - 1] = inf
        # print("sorted_values_list: " + str(sorted_values_list))
        for j in range(2, num - 1):
            # print("j: " + str(j))
            if sorted_values_list[num - 1] - sorted_values_list[0] == 0:
                solutions_distance[i][j] = inf
                continue
            solutions_distance[i][j] += \
                (sorted_values_list[j + 1] - sorted_values_list[j + 1]) / (
                        sorted_values_list[num - 1] - sorted_values_list[0])

    # solutions_distance求平均
    # print("solution_indices: " + str(solution_indices))
    # print("solutions_distance: " + str(solutions_distance))
    for i in range(num):
        avg_distance[i] = (solutions_distance[1][solution_indices[1].index(solution_indices[0][i])]
                           + solutions_distance[0][i]) / 2

    # print("avg_distance: " + str(avg_distance))
    for i in range(num):
        avg_distance_and_indices_dict[solution_indices[0][i]] = avg_distance[i]

    # print("avg_distance_and_indices_dict: " + str(avg_distance_and_indices_dict))
    sorted_tuple_in_list = sorted(avg_distance_and_indices_dict.items(), key=lambda x: x[1], reverse=True)
    # print("sorted_tuple_in_list: " + str(sorted_tuple_in_list))
    indices = [x[0] for x in sorted_tuple_in_list]
    values = [x[1] for x in sorted_tuple_in_list]

    return indices, values


def sort_floor(indices: list, flag: int, population: np.ndarray, labels):
    # sort by value of flag_function
    # func = kkm if flag == 0 else rc
    func = nra if flag == 0 else rc
    indices_nums = len(indices)
    func_value_dict = {}
    for i in range(indices_nums):
        func_value_dict[indices[i]] = func(population[indices[i]], labels[indices[i]])

    sorted_tuple_in_list = sorted(func_value_dict.items(), key=lambda x: x[1], reverse=False)
    # print("sorted_tuple_in_list: " + str(sorted_tuple_in_list))
    indices = [x[0] for x in sorted_tuple_in_list]
    values = [x[1] for x in sorted_tuple_in_list]

    return indices, values


def create_checked_dict(n: list):
    # {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    checked = dict()
    for i in n:
        checked[i] = []

    return checked


def add_num_into_dict(key, value, dictionary):
    tmp_list = dictionary.get(key, [])
    tmp_list.append(value)
    dictionary[key] = tmp_list


def add_list_into_dict(key, nums, dictionary):
    tmp_list = dictionary.get(key, [])
    tmp_list.extend(nums)
    dictionary[key] = copy.deepcopy(tmp_list)


def dfs(adj, start, visited):
    # print(start)
    visited.add(start)
    stack = [[start, 0]]
    while stack:
        (v, next_child_idx) = stack[-1]
        if (v not in adj) or (next_child_idx >= len(adj[v])):
            stack.pop()
            continue
        next_child = adj[v][next_child_idx]
        stack[-1][1] += 1
        if next_child in visited:
            continue
        # print(next_child)
        visited.add(next_child)
        stack.append([next_child, 0])


def add_dict_cross(num1, num2, dictionary):
    # add t1 to dict[t2] and add t2 to dict[t1]
    # {t1: [t2], t2:[t1]}
    tmp_list = dictionary[num1]
    tmp_list.append(num2)
    dictionary[num1] = tmp_list

    tmp_list = dictionary[num2]
    tmp_list.append(num1)
    dictionary[num2] = tmp_list


def index_generator(max_index: int, min_index: int = 0):
    n = min_index
    for i in range(min_index, max_index + 1):
        n += 1
        yield n


def init_population(num, neighbors):
    pigeons = np.zeros((num, shape[0]), dtype=int)
    for i in range(num):
        pigeons[i] = [x[random.randint(0, len(x) - 1)] for x in neighbors]

    return pigeons


FN = ""
P = 0.4
# r = 0.3
tr = 3
archive = list()
# archive_dict = dict()
pigeons_function_value = list()
best_pigeons_function_value = list()
archive_function_value = list()
d_values_sorted = list()
population_num = 0
archive_num = 0
shape = tuple()
inf = 999
elim_threshold = 0.25
dist_d_value = 0.05
f_num = 2
# f_max = [0, 0]  # store the maximums of kkm and rc
# f_min = [100, 100]  # store the minimums of kkm and rc

path = "gml/karate.gml"

graph = get_graph(path)
real_label = np.zeros(shape[0])
archive_array = np.zeros(shape[1], dtype=int)
archive_div = np.zeros(shape[1], dtype=int)
graph_array = matrix2array(graph.get_adjacency())
neighbors_list = get_neighbor_list(graph_array)

best_pigeons = np.zeros(shape[1], dtype=int)
best_pigeons_div = np.zeros(shape[1], dtype=int)

if __name__ == "__main__":
    pio_main(20, 20, "")
