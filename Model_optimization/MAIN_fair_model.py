# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------


# %% Upload Libraries
import pandas as pd
import Ph_PF_run_Concrete as run_pf
import print_results as pr
import numpy as np
import csv
import math
from time import process_time

#run your code
##########################################
# Importing System Data
System_Data_Nodes = pd.read_excel('Nodes_34.xlsx')                          # Node information system (Aggregators - Nominal Demand)
System_Data_Lines = pd.read_excel('Lines_34.xlsx')                          # Branch information system (topology)
System_Time_n_Scale = pd.read_excel('Time_slots.xlsx')                      # Scale factor for nominal demand (Original aggreg. profile) - It can be as many timeslots you need [1, +inf)
System_energy_storage = pd.read_excel('data_storage.xlsx')                      # Scale factor for nominal demand (Original aggreg. profile) - It can be as many timeslots you need [1, +inf)

np.random.seed(30)


##########################################
# Parameters
Vnom = 11       # kV
Snom = 1000     # kVA
Vmin = 0.90     # pu
Vmax = 1.10    # pu
N_MC = 1

cv = 0.0

MC_s = 100       # Number of scenarios
Pct_penetration = 1.0 # Percentage of penetration (P_pv/P_total)


# Loop for MCS
t1_start = process_time()

n = 0
while n < MC_s:
    result, flag = run_pf.run_pf(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, Vmin, Vmax, N_MC, cv,
                           System_energy_storage, Pct_penetration)# System_Data_Flex_max, Vnom, Snom, Vmin, Vmax)

    if flag: # Feasible solution
        print('Completed: {}%'.format(n/MC_s*100))
        pr.print_results(result, n, Pct_penetration, MC_s)
        n += 1
    else:
        n = n    # Re-do solution with other scenario


t1_stop = process_time()

print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)

#############################################

####################################################




