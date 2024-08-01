# Libraries
from timeit import repeat
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from simulator.network import parameter as para
from simulator.node.utils import find_receiver

def q_max_function(q_table, state):
    temp = np.max(q_table, axis=1)
    temp[state] = -np.inf
    return temp

def reward_function(network, mc, q_learning, state, time_stem, fuzzy, receive_func=find_receiver):
    alpha = q_learning.alpha
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, alpha=alpha, fuzzy_controller=fuzzy)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time, receive_func)
    p = get_charge_per_2sec(network, q_learning, state)

    E = np.asarray([network.node[request["id"]].energy for request in q_learning.list_request])
    e = np.asarray([request["avg_energy"] for request in q_learning.list_request])

    p_hat = p / np.sum(p)
    first = np.sum(e * p / E)
    second = nb_target_alive / len(network.target)
    third = np.dot(w, p_hat)

    return first, second, third, charging_time

def init_function(nb_action=81):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time, receive_func=find_receiver):
    p = get_charge_per_2sec(net, q_learning, action_id)
    all_path = get_all_path(net, receive_func)
    time_move = distance.euclidean(q_learning.action_list[mc.state], q_learning.action_list[action_id]) / mc.velocity
    list_dead = [
        request["id"]
        for request_id, request in enumerate(q_learning.list_request)
        if (net.node[request["id"]].energy - time_move * request["avg_energy"]) + 
           (p[request_id] - request["avg_energy"]) * charging_time < 0
    ]

    node_to_weight = {req["id"]: sum(req["id"] in path for path in all_path) 
                      for req in q_learning.list_request}
    
    w = np.array([node_to_weight.get(request["id"], 0) for request in q_learning.list_request])
    total_weight = np.sum(w) + len(w) * 1e-3
    w = (w + 1e-3) / total_weight
    
    nb_target_alive = sum(
        para.base in path and not set(list_dead).intersection(path)
        for path in all_path
    )
    
    return w, nb_target_alive

def get_path(net, sensor_id, receive_func=find_receiver):
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path


def get_charge_per_2sec(net, q_learning, state):
    return np.asarray(
        [2 * para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          q_learning.action_list[state]) + para.beta) ** 2 for
         request in q_learning.list_request])

def get_charging_time(network=None, mc = None, q_learning=None, time_stem=0, state=None, alpha=0.1, fuzzy_controller=None):
    # request_id = [request["id"] for request in network.mc.list_request]
    time_move = distance.euclidean(mc.current, q_learning.action_list[state]) / mc.velocity
    alpha = fuzzy_controller.compute_fuzzy_values(network, q_learning, state)

    energy_min = network.node[0].energy_thresh + alpha * (network.node[0].energy_max - network.node[0].energy_thresh)

    locations = np.array([node.location for node in network.node])
    energies = np.array([node.energy for node in network.node])
    avg_energies = np.array([node.avg_energy for node in network.node])

    action_location = np.array(q_learning.action_list[state])
    distances = np.linalg.norm(locations - action_location, axis=1)
    p = para.alpha / (distances + para.beta) ** 2

    p1 = np.zeros(len(network.node))

    for other_mc in network.mc_list:
        if other_mc.id != mc.id:
            if other_mc.get_status() == "charging":
                d = np.linalg.norm(locations - other_mc.current, axis=1)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - time_stem)
            elif other_mc.get_status() == "moving" and other_mc.state != len(q_learning.q_table) - 1:
                d = np.linalg.norm(locations - other_mc.end, axis=1)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - other_mc.arrival_time)

    s1_mask = (energies - time_move * avg_energies + p1 < energy_min) & (p - avg_energies > 0)
    s2_mask = (energies - time_move * avg_energies + p1 > energy_min) & (p - avg_energies < 0)
    s1_s2_id = np.where(s1_mask | s2_mask)[0]

    t = (energy_min - energies[s1_s2_id] + time_move * avg_energies[s1_s2_id] - p1[s1_s2_id]) / (p[s1_s2_id] - avg_energies[s1_s2_id])

    def calculate_dead_nodes(t_values):
        return np.array([
            np.sum(energies[s1_s2_id] - time_move * avg_energies[s1_s2_id] + p1[s1_s2_id] + (p[s1_s2_id] - avg_energies[s1_s2_id]) * t_val < energy_min)
            for t_val in t_values
        ])

    dead_list = calculate_dead_nodes(t)
    if dead_list.size > 0:
        return t[np.argmin(dead_list)]
    return 0

def network_clustering(optimizer, network=None, nb_cluster=81):
    X = np.array([node.location for node in network.node])
    Y = np.array([node.avg_energy**0.5 for node in network.node])

    Y /= np.linalg.norm(Y)
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X, sample_weight=Y)
    
    charging_pos = [tuple(map(int, pos)) for pos in kmeans.cluster_centers_]
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
        charging_pos.append((int(pos[0]), int(pos[1])))
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
