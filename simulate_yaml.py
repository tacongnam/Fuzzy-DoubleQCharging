import csv
import random
import os
import sys
import copy
import numpy as np
from scipy.stats import sem, t
from tabulate import tabulate
import yaml
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
# from moviepy.editor import ImageSequenceClip
from IPython.display import Video, display

from optimizer.qlearning_kmeans import Q_learningv2
from simulator.mobilecharger.mobilecharger import MobileCharger
from simulator.network.network import Network
from simulator.node.cluster import Cluster
from simulator.node.in_node import InNode
from simulator.node.out_node import OutNode
from simulator.node.sensor_node import SensorNode
from simulator.node.relay_node import RelayNode
from simulator.node.connector_node import ConnectorNode
from simulator.node.const import Node_Type
from simulator.node.target import Target

from simulator.network import parameter as para

class Simulation:
    def __init__(self, file_data):
        with open(file_data, 'r') as file:
            self.net_argc = yaml.safe_load(file)
        self.net_argc = copy.deepcopy(self.net_argc)

        self.target_pos = []

    def makeNetwork(self):        
        self.com_range = self.net_argc['node_phy_spe']['com_range']
        self.sen_range = self.net_argc['node_phy_spe']['sen_range']
        self.prob = self.net_argc['node_phy_spe']['prob_gp']
        self.nb_mc = 3
        self.clusters = 80
        self.package_size = self.net_argc['node_phy_spe']['package_size']
        self.alpha = 0.1
        self.q_alpha = 0.5
        self.q_gamma = 0.5
        self.energy = self.net_argc['node_phy_spe']['capacity']
        self.energy_max = self.net_argc['node_phy_spe']['capacity']
        self.node_pos = self.net_argc['nodes']
        self.energy_thresh = 0.9 * self.energy #net_argc['node_phy_spe']['threshold']  

        self.double_q = True
        #dq = input("Double Q Learning or not? Y / N: ")
        #if dq == "Y":

    def buildSensor(self):
        # Build target
        list_clusters = {}
        target_pos = []
        clusters_data = self.net_argc['Clusters']

        list_clusters[-1] = Cluster(-1, para.base)
        target_id = 0

        for cluster in clusters_data:
            list_clusters[int(cluster['cluster_id'])] = Cluster(int(cluster['cluster_id']),  cluster['centroid'])
            
            for target in cluster['list_targets']:
                new_target = Target(target_id, target, int(cluster['cluster_id']))
                target_pos.append(new_target)
                target_id += 1
        # print(list_clusters)
        # print('Build Sensors - Build targets: Done')
        list_node = []

        # Build connector node
        connector_node_data = self.net_argc['ConnectorNode']

        for node in connector_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = ConnectorNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.CONNECTOR_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build connector node: Done')

        # Build in node
        in_node_data = self.net_argc['InNode']

        for node in in_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = InNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.IN_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            gen_node.init_inNode()
            list_node.append(gen_node)
        
        # print('Build Sensors - Build in node: Done')
        
        # Build out node
        out_node_data = self.net_argc['OutNode']

        for node in out_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = OutNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.OUT_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build out node: Done')
        
        # Build sensor node
        sensor_node_data = self.net_argc['SensorNode']

        for node in sensor_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = SensorNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.SENSOR_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build sensor node: Done')
        
        # Build relay node
        relay_node_data = self.net_argc['RelayNode']

        for node in relay_node_data:
            receive_cluster_id = list_clusters[int(node['receive_cluster_id'])]
            send_cluster_id = list_clusters[int(node['send_cluster_id'])]
            location = node['location']
            gen_node = RelayNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.RELAY_NODE, cluster_id=-1, centroid=None, receive_cluster_id=receive_cluster_id, send_cluster_id=send_cluster_id)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build relay node: Done')

        list_sorted = sorted(list_node, key=lambda x: x.cluster_id, reverse=True)

        return list_sorted, target_pos, list_clusters

    def runSimulator(self, run_times, E_mc):
        try:
            os.makedirs('log')
        except FileExistsError:
            pass
        try:
            os.makedirs('fig')
        except FileExistsError:
            pass
        
        output_file = open("log/q_learning_Kmeans.csv", "w")
        result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
        result.writeheader()
        
        life_time = []

        # Initialize Test case
        para.e_weight = 8
        test_begin = 0
        test_end = 6
        
        for nb_run in range(run_times):
            random.seed(nb_run)

            print("[Simulator] Repeat ", nb_run, ":")
            print("Energy weight: ", para.e_weight)

            # Initialize Sensor Nodes and Targets
            list_node, target_pos, list_clusters = self.buildSensor()

            # Initialize Mobile Chargers
            mc_list = []
            for id in range(self.nb_mc):
                if nb_run < test_begin + 2:
                    mc = MobileCharger(id, energy=E_mc, capacity=E_mc, e_move=1, e_self_charge=540, velocity=5, depot_state = self.clusters, double_q=False)
                    mc_list.append(mc)
                else:
                    mc = MobileCharger(id, energy=E_mc, capacity=E_mc, e_move=1, e_self_charge=540, velocity=5, depot_state = self.clusters, double_q=True)
                    mc_list.append(mc)


            # Construct Network
            net_log_file = "log/network_log_new_network_{}.csv".format(nb_run)
            MC_log_file = "log/MC_log_new_network_{}.csv".format(nb_run)
            experiment = "{}_eweight_{}".format(nb_run, para.e_weight)
            net = Network(list_node=list_node, mc_list=mc_list, target=target_pos, experiment=experiment, com_range=self.com_range, list_clusters=list_clusters)

            # self.PrintOutput(net)
            
            # Initialize Q-learning Optimizer
            q_learning = Q_learningv2(net=net, nb_action=self.clusters, alpha=self.alpha, q_alpha=self.q_alpha, q_gamma=self.q_gamma)

            if nb_run == test_end:
                para.e_weight += 1
                test_begin = test_end + 1
                test_end = test_begin + 6
        
            print("[Simulator] Initializing experiment, repetition {}:\n".format(nb_run))
            print("[Simulator] Network:")
            print(tabulate([['Sensors', len(net.node)], ['Targets', len(net.target)], ['Package Size', self.package_size], ['Sending Freq', self.prob], ['MC', self.nb_mc]], headers=['Parameters', 'Value']), '\n')
            print("[Simulator] Optimizer:")
            print(tabulate([['Alpha', q_learning.q_alpha], ['Gamma', q_learning.q_gamma], ['Theta', q_learning.alpha]], headers=['Parameters', 'Value']), '\n')
        
            # Define log file
            file_name = "log/q_learning_Kmeans_new_network_{}.csv".format(nb_run)
            with open(file_name, "w") as information_log:
                writer = csv.DictWriter(information_log, fieldnames=["time", "nb_dead_node", "nb_package"])
                writer.writeheader()
        
            temp = net.simulate(optimizer=q_learning, t=0, dead_time=0)
            life_time.append(temp[0])
            result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

        confidence = 0.95
        h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
        result.writerow({"nb_run": np.mean(life_time), "lifetime": h, "dead_node": 0})

        return net

    def PrintOutput(self, net):
        plt.figure() 
        image_files = []

        # colors = ['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'maroon', 'olive', 'navy', 'beige', 'indigo', 'turquoise']
        shapes = ['<', '>', 's', 'p', 'x', '*', '^', 'v', 'h', 's', 'p', 'o']
        colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'green', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin']
        path_prefix = "result/images"     
        self.draw(net, shapes, colors)

        # def draw(path_to_save_image):

    def draw(self, net, shapes, colors):
        plt.figure(figsize=(9, 9))
        plt.scatter(para.base[0], para.base[1], marker= "*", color = "purple",s = 200)
        for target in net.target:
            plt.scatter(target.location[0], target.location[1], color="red", marker="*",s = 200)
            # print(net.targets_active)
    
        for node in net.node:
                x = node.location[0]
                y = node.location[1]
                size = 20
                color = colors[node.cluster_id]
                type_node = node.type_node
                if(type_node == 1): marker_ = shapes[0]
                if(type_node == 4):  marker_ = shapes[1]
                if(type_node == 3): marker_ = shapes[2]
                if(type_node == 5):  marker_ = shapes[3]

                if node.level != -1:
                    plt.scatter(x, y, color = color , marker = marker_,s=size)
                else:
                    plt.scatter(x, y, color = 'black' , marker = marker_,s=size * 2)
        relayNodes = [node for node in net.node if node.type_node == Node_Type.RELAY_NODE]
        for node in relayNodes:
            color = 'orange' 
            size = 20
            x = node.location[0]
            y = node.location[1]
            
            plt.scatter(x, y, color = color , marker = 'o',s=size)
    
        plt.show()
    # plt.savefig()

print(r"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
 █████   ███   █████ ███████████    █████████  ██████   █████     █████████   ███                             ████             █████                      
░░███   ░███  ░░███ ░░███░░░░░███  ███░░░░░███░░██████ ░░███     ███░░░░░███ ░░░                             ░░███            ░░███                       
 ░███   ░███   ░███  ░███    ░███ ░███    ░░░  ░███░███ ░███    ░███    ░░░  ████  █████████████   █████ ████ ░███   ██████   ███████    ██████  ████████ 
 ░███   ░███   ░███  ░██████████  ░░█████████  ░███░░███░███    ░░█████████ ░░███ ░░███░░███░░███ ░░███ ░███  ░███  ░░░░░███ ░░░███░    ███░░███░░███░░███
 ░░███  █████  ███   ░███░░░░░███  ░░░░░░░░███ ░███ ░░██████     ░░░░░░░░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███   ███████   ░███    ░███ ░███ ░███ ░░░ 
  ░░░█████░█████░    ░███    ░███  ███    ░███ ░███  ░░█████     ███    ░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███  ███░░███   ░███ ███░███ ░███ ░███     
    ░░███ ░░███      █████   █████░░█████████  █████  ░░█████   ░░█████████  █████ █████░███ █████ ░░████████ █████░░████████  ░░█████ ░░██████  █████    
     ░░░   ░░░      ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░    ░░░░░     ░░░░░░░░░  ░░░░░ ░░░░░ ░░░ ░░░░░   ░░░░░░░░ ░░░░░  ░░░░░░░░    ░░░░░   ░░░░░░  ░░░░░                                                                                                                                                                             
------------------------------------------------------------Qlearning Kmeans Optimization-----------------------------------------------------------------
    
    """)

print("Double Q - all connector - 8x-9x")

p = Simulation('data/hanoi1000n50_allconnect.yaml')
p.makeNetwork()
p.runSimulator(14, 108000)