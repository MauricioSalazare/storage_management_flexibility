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
Vmax = 1.10     # pu
N_MC = 1
cv = 0.2        # Coefficient of variation loads
MC_s = 5       # Number of scenarios
Pct_penetration = 1.1 # Percentage of penetration (P_pv/P_total)


# Loop for MCS
t1_start = process_time()

for n in range(MC_s):
    result = run_pf.run_pf(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, Vmin, Vmax, N_MC, cv, System_energy_storage, Pct_penetration)# System_Data_Flex_max, Vnom, Snom, Vmin, Vmax)

    if n == 0:
        with open('ems_optimization.csv', 'w') as f:
            f.write("Scenario,time,")
            for i in result.Ob:
                f.write("v_%d," % i)
            f.write("Loading, storage_P, SOC")
            f.write("\n")

            for t in result.OT:
                # for s in result.Os:
                f.write("%d,%d," % (n, t))
                for i in result.Ob:
                    f.write("%.6f," % (math.sqrt(result.V[i, t, 0].value)))
                f.write("%.6f," % (math.sqrt(result.I[1, 2, t, 0].value)/result.Imax[1,2].value*100))
                for b in result.Ost:
                    f.write("%.6f, %.6f" % (result.Pess[b, t, 0].value*result.Snom.value, result.SOC[b, t, 0].value*100))
                f.write("\n")

        f.close()

    else:
        with open('ems_optimization.csv', 'a') as f:
            # f.write("\n")
            for t in result.OT:
                # for s in result.Os:
                f.write("%d,%d," % (n, t))
                for i in result.Ob:
                    f.write("%.6f," % (math.sqrt(result.V[i, t, 0].value)))
                f.write("%.6f," % (math.sqrt(result.I[1, 2, t, 0].value)/result.Imax[1,2].value*100))
                for b in result.Ost:
                    f.write("%.6f, %.6f" % (result.Pess[b, t, 0].value*result.Snom.value, result.SOC[b, t, 0].value*100))
                f.write("\n")
        f.close()

t1_stop = process_time()

print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)

#############################################

####################################################

# pr.print_results(result)



