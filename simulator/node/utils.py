import numpy as np
from scipy.spatial import distance

from simulator.network import parameter as para


def to_string(node):
    """
    print information of a node
    :param node: sensor node
    :return: None
    """
    print("Id =", node.id, "Location =", node.location, "Energy =", node.energy, "ave_e =", node.avg_energy,
          "Neighbor =", node.neighbor)


# def find_receiver(node, net):
#     """
#     find receiver node
#     :param node: node send this package
#     :param net: network
#     :return: find node nearest base from neighbor of the node and return id of it
#     """
#     if not node.is_active:
#         return -1
#     list_d = [distance.euclidean(para.base, net.node[neighbor_id].location) if net.node[
#         neighbor_id].is_active else float("inf") for neighbor_id in node.neighbor]
#     id_min = np.argmin(list_d)
#     if distance.euclidean(node.location, para.base) <= list_d[id_min]:
#         return -1
#     else:
#         return node.neighbor[id_min]


def find_receiver(node, net):
    """
    find receiver node
    :param node: node send this package
    :param net: network
    :return: find node nearest base from neighbor of the node and return id of it
    """
    if not node.is_active:
        return -1
    candidate = [neighbor for neighbor in node.neighbor if
                 neighbor.level < node.level and neighbor.is_active]
    if candidate:
        d = [distance.euclidean(node.location, para.base) for node in candidate]
        id_min = np.argmin(d)
        return candidate[id_min]
    else:
        return -1


def request_function(node, index, optimizer, t):
    """
    add a message to request list of mc.
    :param node: the node request
    :param mc: mobile charger
    :param t: time get request
    :return: None
    """
    optimizer.list_request.append(
        {"id": index, "energy": node.energy, "avg_energy": node.avg_energy, "energy_estimate": node.energy,
         "time": t})


def estimate_average_energy(node):
    """
    function to estimate average energy
    user can replace with other function
    :return: a scalar which is calculated from check point list
    """
    return node.check_point[-1]["avg_e"]
