# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------

# %% Upload Libraries
from pyomo.environ import *
import create_1_ph_pf_model as pf
import math
import pandas as pd
import numpy
import processing_data_network as dn


def run_pf(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, Vmin, Vmax, N_MC, cv,
           System_energy_storage, Pct_penetration):
    # Preparing System Data for Pyomo
    Data_Network = dn.processing_system_data_for_pyomo(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, N_MC, cv,
                                                       System_energy_storage, Pct_penetration)

    # Create the Model
    model = pf.create_1_ph_pf_model(Vnom, Snom, Vmin, Vmax, Data_Network)

    # Define the Solver
    solver = SolverFactory('ipopt')  # couenne
    solver.options['print_level'] = '0'


    # Solve
    solver.solve(model, tee=True)

    #####################################################################################
    #####################################################################################

    #


    return model
