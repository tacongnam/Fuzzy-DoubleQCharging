import numpy as np
from scipy.spatial import distance

from optimizer.utils import init_function, q_max_function, reward_function, network_clustering, network_clustering_v2
from simulator.node.utils import find_receiver


class Q_learningv2:
    def __init__(self, init_func=init_function, nb_action=80, alpha=0, q_alpha=0.5, q_gamma=0.5, load_checkpoint=False):
        self.action_list = []
        self.nb_action = nb_action
        self.q_table = init_func(nb_action=nb_action)
        self.q1 = init_func(nb_action=nb_action)
        self.q2 = init_func(nb_action=nb_action)
        # self.state = nb_action
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        self.alpha = alpha
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma

    def update(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function, reward_func=reward_function):
        if not len(self.list_request):
            return self.action_list[mc.state], -1.0
        
        self.set_reward(q_table=self.q_table, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
    
        '''
        if np.random.rand() < 0.5:
            self.set_reward(q_table=self.q1, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
            self.q1[mc.state] =  (1 - self.q_alpha) * self.q1[mc.state] + self.q_alpha * (
                self.reward + self.q_gamma * self.q_max(mc, self.q2, q_max_func))
            self.choose_next_state(mc, self.q1, network)
        else:
            self.set_reward(q_table=self.q2, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
            self.q2[mc.state] =  (1 - self.q_alpha) * self.q2[mc.state] + self.q_alpha * (
                self.reward + self.q_gamma * self.q_max(mc, self.q1, q_max_func))
            self.choose_next_state(mc, self.q2, network)
        '''

        self.q_table[mc.state] = (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (
               self.reward + self.q_gamma * self.q_max(mc=mc, table=self.q_table, q_max_func=q_max_func))

        self.choose_next_state(mc, self.q_table, network)
        
        if mc.state == len(self.action_list) - 1:
            charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
        else:
            charging_time = self.charging_time[mc.state]
        
        if charging_time > 1:
            print("[Optimizer] MC #{} is sent to point {} (id={}) and charge for {:.2f}s".format(mc.id, self.action_list[mc.state], mc.state, charging_time))

        # print(self.charging_time)
        return self.action_list[mc.state], charging_time

    def q_max(self, mc, table, q_max_func=q_max_function):
        return q_max_func(q_table=table, state=mc.state)

    def set_reward(self, q_table, mc = None, time_stem=0, reward_func=reward_function, network=None):
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(q_table):
            temp = reward_func(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third
        print(self.reward)
        self.reward_max = list(zip(first, second, third))

    def choose_next_state(self, mc, table, network):
        # next_state = np.argmax(self.q_table[mc.state])
        if mc.energy < 10:
            mc.state = len(table) - 1
            print('[Optimizer] MC #{} energy is running low ({:.2f}), and needs to rest!'.format(mc.id, mc.energy))
        else:
            mc.state = np.argmax(table[mc.state])
            # print(self.reward_max[mc.state])
            # print(self.action_list[mc.state])
    
    def net_partition(self, net=None, net_clustering_func=network_clustering):
        self.action_list = net_clustering_func(self, network=net, nb_cluster=self.nb_action)
