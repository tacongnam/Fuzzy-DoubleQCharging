import sys
from Simulation import Simulation
import numpy as np
import time

from simulator.network import parameter as para

def run_weight(begin: float, end: float, testcase):
    print("Test weight")
    print(f"begin: {begin}, end: {end}")

    for q_a in np.arange(0.1, 0.9):
        for q_b in np.arange(0.1, 0.9):
            for q_c in np.arange(0.1, 0.9):
                print(f"{q_a}, {q_b}, {q_c}")
                p = Simulation(f'data/{testcase}')
                p.makeNetwork(0.5, 0.5)

                para.e_weight_a = q_a
                para.e_weight_b = q_b
                para.e_weight_c = q_c

                net = p.runSimulator(1, 54000, 1, 1)

                print(net.t)

def main():
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
     
    
    if len(sys.argv) != 5   :
        print("Usage: python simulate_yaml.py <type> <param1> <param2> <param3>")
        sys.exit(1)
    else:
        print("OK!")
    
    run_type = sys.argv[1]
    param1 = float(sys.argv[2])
    param2 = float(sys.argv[3])
    param3 = sys.argv[4]

    if run_type == 'weight':
        run_weight(param1, param2, param3)
    # elif run_type == 'co-formula':
    #     run_coefficient(param1, param2, param3)
    # elif run_type == 'theta':
    #     run_theta(param1, param2, param3)


if __name__ == "__main__":
    main()