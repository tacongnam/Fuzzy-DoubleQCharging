# Libraries
from timeit import repeat
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Modules
from simulator.network import parameter as para
from simulator.node.utils import find_receiver

BASE = -1

def q_max_function(q_table, state):
    temp = [max(row) for index, row in enumerate(q_table)]
    return np.asarray(temp)


def reward_function(network, mc, q_learning, state, time_stem):
    alpha = q_learning.alpha
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, alpha=alpha)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time)
    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)
    E = np.asarray([network.node[request["id"]].energy for request in q_learning.list_request])
    e = np.asarray([request["avg_energy"] for request in q_learning.list_request])
    second = nb_target_alive / len(network.target)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    return first, second, third, charging_time

def init_function(nb_action=81):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time):
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = q_learning.all_path
    time_move = distance.euclidean(q_learning.action_list[mc.state], q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in q_learning.list_request]

    for request_id, request in enumerate(q_learning.list_request):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(net.node[request["id"]].id)

    for request_id, request in enumerate(q_learning.list_request):
        nb_path = 0
        for path in all_path:
            if net.node[request["id"]].id in path:
                nb_path += 1
        w[request_id] = nb_path

    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    
    for path in all_path:
        if BASE in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_path(net, sensor):
    path = [sensor.id]
    if distance.euclidean(sensor.location, para.base) <= sensor.com_ran:
        path.append(-1)
    else:
        receive = sensor.find_receiver(net=net)
        if receive.id != -1:
            path.extend(get_path(net, receive))
    return path


def get_all_path(net):
    list_path = []

    for target in net.target:
        new_path = []
        for node in target.listSensors:
            new_path = get_path(net, node[0])
            if BASE in new_path:
                break
        list_path.append(new_path)

    return list_path


def get_charge_per_sec(net, q_learning, state):
    return np.asarray(
        [para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          q_learning.action_list[state]) + para.beta) ** 2 for
         request in q_learning.list_request])

def FLCDS_model(network=None):
    max_energy = network.node[0].energy_max
    
    E_min = ctrl.Antecedent(np.linspace(0, max_energy, num = 1001), 'E_min')
    L_r = ctrl.Antecedent(np.arange(0, len(network.node) + 1), 'L_r')
    Theta = ctrl.Consequent(np.linspace(0, 1, num = 101), 'Theta')

    L_r['L'] = fuzz.trapmf(L_r.universe, [0, 0, 2, 6])
    L_r['M'] = fuzz.trimf(L_r.universe, [2, 6, 10])
    L_r['H'] = fuzz.trapmf(L_r.universe, [6, 10, len(network.node), len(network.node)])

    E_min['L'] = fuzz.trapmf(E_min.universe, [0, 0, 0.25 * max_energy, 0.5 * max_energy])
    E_min['M'] = fuzz.trimf(E_min.universe, [0.25 * max_energy, 0.5 * max_energy, 0.75 * max_energy])
    E_min['H'] = fuzz.trapmf(E_min.universe, [0.5 * max_energy, 0.75 * max_energy, max_energy, max_energy])

    Theta['VL'] = fuzz.trimf(Theta.universe, [0, 0, 1/3])
    Theta['L'] = fuzz.trimf(Theta.universe, [0, 1/3, 2/3])
    Theta['M'] = fuzz.trimf(Theta.universe, [1/3, 2/3, 1])
    Theta['H'] = fuzz.trimf(Theta.universe, [2/3, 1, 1])

    R1 = ctrl.Rule(L_r['L'] & E_min['L'], Theta['H'])
    R2 = ctrl.Rule(L_r['L'] & E_min['M'], Theta['M'])
    R3 = ctrl.Rule(L_r['L'] & E_min['H'], Theta['L'])
    R4 = ctrl.Rule(L_r['M'] & E_min['L'], Theta['M'])
    R5 = ctrl.Rule(L_r['M'] & E_min['M'], Theta['L'])
    R6 = ctrl.Rule(L_r['M'] & E_min['H'], Theta['VL'])
    R7 = ctrl.Rule(L_r['H'] & E_min['L'], Theta['L'])
    R8 = ctrl.Rule(L_r['H'] & E_min['M'], Theta['VL'])
    R9 = ctrl.Rule(L_r['H'] & E_min['H'], Theta['VL'])

    FLCDS_ctrl = ctrl.ControlSystem([R1, R2, R3,
                             R4, R5, R6,
                             R7, R8, R9])
    FLCDS = ctrl.ControlSystemSimulation(FLCDS_ctrl)

    return FLCDS

def get_charging_time(network=None, mc = None, q_learning=None, time_stem=0, state=None, alpha=0.1):
    time_move = distance.euclidean(mc.current, q_learning.action_list[state]) / mc.velocity

    # request_id = [request["id"] for request in network.mc.list_request]
    FLCDS = q_learning.FLCDS    
    L_r_crisp = len(q_learning.list_request)
    E_min_crisp = network.node[network.find_min_node()].energy

    FLCDS.input['L_r'] = L_r_crisp
    FLCDS.input['E_min'] = E_min_crisp
    FLCDS.compute()
    alpha = FLCDS.output['Theta']
    q_learning.alpha = alpha

    # energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max
    energy_min = network.node[0].energy_thresh + alpha * (network.node[0].energy_max - network.node[0].energy_thresh)

    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for node in network.node:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in network.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - time_stem)
            elif other_mc.id != mc.id and other_mc.get_status() == "moving" and other_mc.state != len(q_learning.q_table) - 1:
                d = distance.euclidean(other_mc.end, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - other_mc.arrival_time)
        
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node, p, p1))
    
    t = []
    for node, p, p1 in s1:
        t.append((energy_min - node.energy + time_move * node.avg_energy - p1) / (p - node.avg_energy))
    for node, p, p1 in s2:
        t.append((energy_min - node.energy + time_move * node.avg_energy - p1) / (p - node.avg_energy))
    
    dead_list = [] 
    for item in t:
        nb_dead = 0
        for node, p, p1 in s1:
            temp = node.energy - time_move * node.avg_energy + p1 + (p - node.avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for node, p, p1 in s2:
            temp = node.energy - time_move * node.avg_energy + p1 + (p - node.avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    return 0

def network_clustering(optimizer, network=None, nb_cluster=81):
    X = []
    Y = []
    for node in network.node:
        node.set_check_point(200)
        X.append(node.location)
        Y.append(node.avg_energy**0.5)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    d = np.linalg.norm(Y)
    # print(Y, d)
    Y = Y/d
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X, sample_weight=Y)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append([int(pos[0]), int(pos[1])])
    charging_pos.append(para.depot)
    # print(charging_pos, file=open('log/centroid.txt', 'w'))
    node_distribution_plot(network=network, charging_pos=charging_pos)
    network_plot(network=network, charging_pos=charging_pos)
    return charging_pos

def network_clustering_v2(optimizer, network=None, nb_cluster=81):
    X = []
    Y = []
    min_node = 1000
    for node in network.node:
        node.set_check_point(200)
        if node.avg_energy != 0:
            min_node = min(min_node, node.avg_energy)
    for node in network.node:
        repeat = int(node.avg_energy/min_node)
        for _ in range(repeat):
            X.append(node.location)
            Y.append(node.avg_energy)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    d = np.linalg.norm(Y)
    Y = Y/d
    # print(d)
    # print(Y)
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append([int(pos[0]), int(pos[1])])
    charging_pos.append(para.depot)
    # print(charging_pos, file=open('log/centroid.txt', 'w'))
    # node_distribution_plot(network=network, charging_pos=charging_pos)
    network_plot(network=network, charging_pos=charging_pos)
    return charging_pos
    
def node_distribution_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.node:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    plt.hist(c_node, bins=100)
    plt.savefig('fig/node_distribution.png')

def network_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.node:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    for centroid in charging_pos:
        x_centroid.append(centroid[0])
        y_centroid.append(centroid[1])
    c_node = np.array(c_node)
    d = np.linalg.norm(c_node)
    c_node = c_node / d * 80
    plt.scatter(x_node, y_node, s = c_node)
    plt.scatter(x_centroid, y_centroid, c='red', marker='^')
    plt.savefig('fig/network_plot.png')
