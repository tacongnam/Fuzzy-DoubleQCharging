import sys
from Simulation import Simulation
import numpy as np
import time

from simulator.network import parameter as para

def run_weight(a1, a2, b1, b2, c1, c2, testcase):
    print("Test weight")
    print(f"Energy - begin: {a1}, end: {a2}")
    print(f"Connection - begin: {b1}, end: {b2}")
    print(f"Covering - begin: {c1}, end: {c2}")

    for q_a in np.arange(a1, a2, 1):
        for q_b in np.arange(b1, b2, 1):
            for q_c in np.arange(c1, c2, 1):
                print(f"{q_a}, {q_b}, {q_c}")
                p = Simulation(f'data/{testcase}')
                p.makeNetwork(0.5, 0.5)

                para.e_weight_a = q_a
                para.e_weight_b = q_b
                para.e_weight_c = q_c

                net = p.runSimulator(1, 54000, 1, 1)

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
     
    
    run_type = sys.argv[1]
    param1 = float(sys.argv[2])
    param2 = float(sys.argv[3])
    param3 = float(sys.argv[4])
    param4 = float(sys.argv[5])
    param5 = float(sys.argv[6])
    param6 = float(sys.argv[7])
    param7 = sys.argv[8]

    if run_type == 'weight':
        run_weight(param1, param2, param3, param4, param5, param6, param7)
    # elif run_type == 'co-formula':
    #     run_coefficient(param1, param2, param3)
    # elif run_type == 'theta':
    #     run_theta(param1, param2, param3)


if __name__ == "__main__":
    main()