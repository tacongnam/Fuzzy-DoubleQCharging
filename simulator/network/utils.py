import random
import pickle
import numpy as np
from scipy.spatial import distance

from simulator.network.package import Package
from simulator.node.const import Node_Type

def uniform_com_func(net):
    for target in net.target:
        # Xac xuat truyen goi tin moi giay = 60%
        if random.random() > 0.6:
            continue
        for node in target.listSensors:
            if node[0].is_active == False:
                continue
            #temp_package = Package(is_energy_info=True)
            #node[0].send(net, temp_package, receiver=node[0].find_receiver(net))

            #if temp_package.path[-1] == -1:
            package = Package(is_energy_info=False)
            node[0].send(net, package, receiver=node[0].find_receiver(net))
            #print("Sent success target {} from node {}, path {}".format(target.location, node[0].id, package.path))
            break
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
    for target in net.target:
        for node in target.listSensors:
            if node[0].is_active == False:
                continue

            temp_package = Package(is_energy_info=True)
            node[0].send(net, temp_package, receiver=node[0].find_receiver(net))

            if temp_package.path[-1] == -1:
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