import random
import pickle
import numpy as np
from scipy.spatial import distance

from simulator.network.package import Package
from simulator.node.const import Node_Type

def uniform_com_func(net):
    sent_target = np.zeros(len(net.target), dtype = bool)
    
    for node in net.node:
        for id in range(len(net.target)):
            if sent_target[id] == False and distance.euclidean(node.location, net.target[id].location) <= node.sen_ran:
                temp_package = Package(is_energy_info=True)
                node.send(net, temp_package, receiver=node.find_receiver(net))

                if temp_package.path[-1] == -1:
                    package = Package(is_energy_info=False)
                    node.send(net, package, receiver=node.find_receiver(net))
                    sent_target[id] = True
    return True


def to_string(net):
    min_energy = 10 ** 10
    min_node = -1
    for node in net.node:
        if node.energy < min_energy:
            min_energy = node.energy
            min_node = node
    min_node.print_node()

def count_package_function(net):
    count = 0

    sent_target = np.zeros(len(net.target), dtype = bool)

    for id in range(len(net.target)):
        for node in net.node:
            if distance.euclidean(node.location, net.target[id].location) <= node.sen_ran:
                package = Package(is_energy_info=True)
                node.send(net, package, receiver=node.find_receiver(net))

                if package.path[-1] == -1:
                    count += 1
                    break
    return count

def set_checkpoint(t=0, network=None, optimizer=None, dead_time=0):
    nb_run = int(network.experiment.split('_')[0])
    checkpoint = {
        'time'              : t,
        'experiment_type'   : 'new network',
        'nb_run'            : nb_run,
        'network'           : network,
        'optimizer'         : optimizer,
        'dead_time'         : dead_time
    }
    with open('checkpoint/checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    print("[Simulator] Simulation checkpoint set at {}s".format(t))