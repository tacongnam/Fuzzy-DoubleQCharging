import math
from scipy.spatial import distance
import numpy as np

from simulator.node.const import Node_Type
from simulator.node.node import Node
from simulator.network import parameter as para

class ConnectorNode(Node):  
    def find_receiver(self, net):
        """
        find receiver node
        :param node: node send this package
        :param net: network
        :return: find node nearest base from neighbor of the node and return id of it
        """
        if not self.is_active:
            return Node(id = -1)
        
        distance_min = 10000007
        node_min = Node(id = -1)
            
        for node in self.neighbor:
            if node.is_active == True and (node.type_node in [Node_Type.IN_NODE, Node_Type.OUT_NODE, Node_Type.CONNECTOR_NODE] 
                and node.cluster_id == self.cluster_id and self.level > node.level): 
            # if(name == "ConnectorNode" or name == "InNode" or name == "OutNode") and self.cluster_id == node.cluster_id:
            # với connector node thì cần phải xét level nếu không sẽ gửi lộn

                if distance.euclidean(node.location, net.listClusters[self.cluster_id].centroid) < distance_min:
                    node_min = node
                    distance_min = distance.euclidean(node.location, self.location)
        
        return node_min
    
    def probe_neighbors(self, network):
        self.neighbor.clear()
        self.potentialSender.clear()

        for node in network.node:
            if node.is_active == True and self != node and distance.euclidean(node.location, self.location) <= self.com_ran:
                self.neighbor.append(node)
                if(node.type_node in [Node_Type.SENSOR_NODE, Node_Type.CONNECTOR_NODE] and self.cluster_id == node.cluster_id):
                    self.potentialSender.append(node)