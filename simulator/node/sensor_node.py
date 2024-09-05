import math
from scipy.spatial import distance
import numpy as np

from simulator.node.const import Node_Type
from simulator.node.node import Node
from simulator.network import parameter as para

class SensorNode(Node):  
    def find_receiver(self, net):
        """
        find receiver node
        :param net: network
        :return: find node nearest base from neighbor of the node and return id of it
        """
        if not self.is_active:
            return Node(id = -1)
        
        if self.candidate is not None and self.candidate.is_active == True:
            return self.candidate
        
        distance_min = 10000007.0
        node_min = Node(id = -1)

        for node in self.neighbor:
            if  node.is_active == True and (node.type_node in [Node_Type.IN_NODE, Node_Type.OUT_NODE]):
                self.candidate = node
                return node
            
            if  node.is_active == True and (node.type_node == Node_Type.CONNECTOR_NODE and node.cluster_id == self.cluster_id):
                if distance.euclidean(node.location, net.listClusters[self.cluster_id].centroid) < distance_min:
                    self.candidate = node
                    node_min = node
                    distance_min = distance.euclidean(node.location, net.listClusters[self.cluster_id].centroid)
        
        return node_min
    
    def probe_neighbors(self, network):
        self.neighbor.clear()
        self.potentialSender.clear()

        for node in network.node:
            if  node.is_active == True and self != node and distance.euclidean(node.location, self.location) <= self.com_ran:
                self.neighbor.append(node)
                self.potentialSender.append(node)