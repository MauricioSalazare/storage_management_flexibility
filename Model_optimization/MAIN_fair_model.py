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


##########################################
# Importing System Data
System_Data_Nodes = pd.read_excel('Nodes_34.xlsx')                          # Node information system (Aggregators - Nominal Demand)
System_Data_Lines = pd.read_excel('Lines_34.xlsx')                          # Branch information system (topology)
System_Time_n_Scale = pd.read_excel('Time_slots.xlsx')                      # Scale factor for nominal demand (Original aggreg. profile) - It can be as many timeslots you need [1, +inf)

np.random.seed(30)

##########################################
# Parameters
Vnom = 11       # kV
Snom = 1000     # kVA
Vmin = 0.85     # pu
Vmax = 1.00     # pu
N_MC = 1       # Number of scenarios
cv = 0.1        # Coefficient of variation

result = run_pf.run_pf(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, Vmin, Vmax, N_MC, cv)# System_Data_Flex_max, Vnom, Snom, Vmin, Vmax)

#############################################

Tot_Cost = (sum(sum(result.cf[i, t].value for i in result.Ob.value) for t in result.OT.value))*result.Snom.value # Returs total cost of flexibility ( sum(c_i(K_{i,t}, L_{i,t})) )
Cost_max = result.obj()*result.Snom.value   # Cost of maximum c_i --> Objective function
V = result.V                                # use V[i,t].value to get specific voltage results
I = result.I
# K = result.K

with open('ems_optimization.csv', 'w') as f:
    f.write("Scenario,time,")
    for i in result.Ob:
        f.write("v_%d,"%i)
    f.write("\n")
    for t in result.OT:
        for s in result.Os:
            f.write("%d,%d," % (s, t))
            for i in result.Ob:
                f.write("%.6f,"% (math.sqrt(result.V[i, t, s].value)))
            f.write("\n")





        # print('{:<8d}'.format(i), end=" ")
        # for t in model.OT:
        #     print('{:>16.6f}'.format(math.sqrt(model.V[i, t].value)), end=" ")
        # print('')
####################################################

# pr.print_results(result)



