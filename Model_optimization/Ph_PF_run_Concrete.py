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
    results = solver.solve(model, tee=True)


    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        flag = True
    # Do something when the solution in optimal and feasible
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('Infeasible: Problem solved once again')
        flag = False
    # Do something when model in infeasible
    else:
        # Something else is wrong
        print("Solver Status: ", result.solver.status)
        flag = False

    #####################################################################################
    #####################################################################################

    #


    return model, flag
