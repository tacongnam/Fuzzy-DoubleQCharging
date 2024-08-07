import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

class FuzzyController:
    def __init__(self, network):
        # Define fuzzy variables and their ranges
        E_min = ctrl.Antecedent(np.linspace(0, 10, num=1001), 'E_min')
        L_r = ctrl.Antecedent(np.arange(0, len(network.node) + 1), 'L_r')
        Theta = ctrl.Consequent(np.linspace(0, 1, num=101), 'Theta')
        
        # Define fuzzy membership functions
        L_r['L'] = fuzz.trapmf(L_r.universe, [0, 0, 2, 6])
        L_r['M'] = fuzz.trimf(L_r.universe, [2, 6, 10])
        L_r['H'] = fuzz.trapmf(L_r.universe, [6, 10, len(network.node), len(network.node)])
        
        E_min['L'] = fuzz.trapmf(E_min.universe, [0, 0, 2.5, 5])
        E_min['M'] = fuzz.trimf(E_min.universe, [2.5, 5.0, 7.5])
        E_min['H'] = fuzz.trapmf(E_min.universe, [5, 7.5, 10, 10])
        
        Theta['VL'] = fuzz.trimf(Theta.universe, [0, 0, 1/3])
        Theta['L'] = fuzz.trimf(Theta.universe, [0, 1/3, 2/3])
        Theta['M'] = fuzz.trimf(Theta.universe, [1/3, 2/3, 1])
        Theta['H'] = fuzz.trimf(Theta.universe, [2/3, 1, 1])
        
        # Define fuzzy rules
        rules = [
            ctrl.Rule(L_r['L'] & E_min['L'], Theta['H']),
            ctrl.Rule(L_r['L'] & E_min['M'], Theta['M']),
            ctrl.Rule(L_r['L'] & E_min['H'], Theta['L']),
            ctrl.Rule(L_r['M'] & E_min['L'], Theta['M']),
            ctrl.Rule(L_r['M'] & E_min['M'], Theta['L']),
            ctrl.Rule(L_r['M'] & E_min['H'], Theta['VL']),
            ctrl.Rule(L_r['H'] & E_min['L'], Theta['L']),
            ctrl.Rule(L_r['H'] & E_min['M'], Theta['VL']),
            ctrl.Rule(L_r['H'] & E_min['H'], Theta['VL']),
        ]
        
        # Create fuzzy control system and simulation
        self.FLCDS_ctrl = ctrl.ControlSystem(rules)
        self.FLCDS = ctrl.ControlSystemSimulation(self.FLCDS_ctrl)

    def compute_fuzzy_values(self, network, q_learning, state):
        # Get inputs for fuzzy logic
        L_r_crisp = len(q_learning.list_request)
        E_min_crisp = network.node[network.find_min_node()].energy
        
        # Set inputs and compute output
        self.FLCDS.input['L_r'] = L_r_crisp
        self.FLCDS.input['E_min'] = E_min_crisp
        self.FLCDS.compute()
        
        alpha = self.FLCDS.output['Theta']
        q_learning.alpha = alpha

        return alpha