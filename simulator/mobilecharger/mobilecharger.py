import csv
from scipy.spatial import distance
import numpy as np
from simulator.mobilecharger.utils import get_location, charging
from simulator.network import parameter as para


class MobileCharger:
    def __init__(self, id,  energy=None, e_move=None, start=para.depot, end=para.depot, velocity=None,
                 e_self_charge=None, capacity=None, depot_state=80):
        self.id = id
        self.is_stand = False  # is true if mc stand and charge
        self.is_self_charge = False  # is true if mc is charged
        self.is_active = False

        self.start = start  # from location
        self.end = end  # to location
        self.current = start  # location now
        self.end_time = -1
        self.moving_time = 0
        self.arrival_time = 0

        self.energy = energy  # energy now
        self.capacity = capacity  # capacity of mc
        self.e_move = e_move  # energy for moving
        self.e_self_charge = e_self_charge  # energy receive per second
        self.velocity = velocity  # velocity of mc
        self.depot_state = depot_state
        self.state = depot_state # Current state in Q_table

    def get_status(self):
        if not self.is_active:
            return "deactivated"
        if not self.is_stand:
            return "moving"
        if not self.is_self_charge:
            return "charging"
        return "self_charging" 

    def update_location(self, func=get_location):
        self.current = func(self)
        self.energy -= 2 * self.e_move

    def charge(self, net=None, node=None, func=charging):
        func(self, net, node)

    def self_charge(self):
        self.energy = min(self.energy + 2 * self.e_self_charge, self.capacity)

    def check_state(self):
        current = np.array(self.current)
        end = np.array(self.end)
        depot = np.array(para.depot)
    
        distance_to_end = distance.euclidean(current, end)
        distance_to_depot = distance.euclidean(end, depot)
    
        self.is_stand = distance_to_end < 1
        self.is_self_charge = distance_to_depot < 1e-3

        if self.is_stand:
            self.current = self.end

    def get_next_location(self, network, time_stem, optimizer=None):
        next_location, charging_time = optimizer.update(self, network, time_stem)
        self.start = self.current
        self.end = next_location
        distance_move = distance.euclidean(self.start, self.end)
        self.moving_time = distance_move / self.velocity
        self.end_time = time_stem + self.moving_time + charging_time
        self.arrival_time = time_stem + self.moving_time

        print(f"[Mobile Charger] MC #{self.id} moves to {self.end} in {self.moving_time:.2f}s and charges for {charging_time:.2f}s")

        mc_info = {
            'time_stamp': time_stem,
            'id': self.id,
            'starting_point': self.start,
            'destination_point': self.end,
            'decision_id': self.state,
            'charging_time': charging_time,
            'moving_time': self.moving_time
        }

        with open(network.mc_log_file, "a", newline='') as mc_log_file:
            writer = csv.DictWriter(mc_log_file, fieldnames=mc_info.keys())
            if mc_log_file.tell() == 0:  # Nếu file trống, viết tiêu đề
                writer.writeheader()
            writer.writerow(mc_info)

    def run(self, network, time_stem, net=None, optimizer=None):
        # print(self.energy, self.start, self.end, self.current)
        if ((not self.is_active) and optimizer.list_request) or abs(time_stem - self.end_time) < 1:
            self.is_active = True

            # Lọc yêu cầu không còn hợp lệ
            node_energies = np.array([net.node[req["id"]].energy for req in optimizer.list_request])
            node_thresholds = np.array([net.node[req["id"]].energy_thresh for req in optimizer.list_request])
            valid_requests = node_energies < node_thresholds

            optimizer.list_request = [req for req, valid in zip(optimizer.list_request, valid_requests) if valid]
            for req, valid in zip(optimizer.list_request, valid_requests):
                if not valid:
                    net.node[req["id"]].is_request = False

            if not optimizer.list_request:
                self.is_active = False

            self.get_next_location(network=network, time_stem=time_stem, optimizer=optimizer)
        else:
            if self.is_active:
                if not self.is_stand:
                    # print("moving")
                    self.update_location()
                elif not self.is_self_charge:
                    # print("charging")
                    self.charge(net)
                else:
                    # print("self charging")
                    self.self_charge()

        if self.energy < para.E_mc_thresh and not self.is_self_charge and self.end != para.depot:
            self.start = self.current
            self.end = para.depot
            self.is_stand = False
            charging_time = self.capacity / self.e_self_charge
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()
