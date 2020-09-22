# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------


from pyomo.environ import *
import create_1_ph_pf_model as pf
import math
import pandas as pd
import numpy as np

def processing_system_data_for_pyomo(System_Data_Nodes, System_Data_Lines, System_Time_n_Scale, Vnom, Snom, N_MC, cv,
                                     System_energy_storage, Pct_penetration):# System_Data_Flex_max, Vnom, Snom):

    # Network Data
    Ob = [System_Data_Nodes.loc[i, 'NODES'] for i in System_Data_Nodes.index]
    Ost = [System_energy_storage.loc[i, 'Bus'] for i in System_energy_storage.index]
    Os = range(N_MC)


    Tb = {Ob[i]: System_Data_Nodes.loc[i, 'Tb'] for i in System_Data_Nodes.index}
    PD = {Ob[i]: System_Data_Nodes.loc[i, 'PDn'] / Snom for i in System_Data_Nodes.index}
    QD = {Ob[i]: System_Data_Nodes.loc[i, 'QDn'] / Snom for i in System_Data_Nodes.index}
    # Ppv = {Ob[i]: Pct_penetration*System_Data_Nodes.loc[i, 'PVn'] / Snom for i in System_Data_Nodes.index}


    # SOCini = {Ost[i] : System_energy_storage.loc[i, 'SOCini'] /100 for i in System_energy_storage.index}
    SOCM = {Ost[i] : System_energy_storage.loc[i, 'SOCM'] /100 for i in System_energy_storage.index}
    SOCm = {Ost[i] : System_energy_storage.loc[i, 'SOCm'] /100 for i in System_energy_storage.index}
    PmaxE = {Ost[i] : System_energy_storage.loc[i, 'PmaxE'] /Snom for i in System_energy_storage.index}
    EC = {Ost[i] : System_energy_storage.loc[i, 'EC'] /Snom for i in System_energy_storage.index}
    Cess = {Ost[i] : System_energy_storage.loc[i, 'Cess']*Snom for i in System_energy_storage.index}



    Ol = {(System_Data_Lines.loc[i, 'FROM'], System_Data_Lines.loc[i, 'TO']) for i in System_Data_Lines.index}
    R = {(System_Data_Lines.loc[i, 'FROM'], System_Data_Lines.loc[i, 'TO']): System_Data_Lines.loc[i, 'R'] / (
                Vnom ** 2 * 1000 / (Snom)) for i in System_Data_Lines.index}
    X = {(System_Data_Lines.loc[i, 'FROM'], System_Data_Lines.loc[i, 'TO']): System_Data_Lines.loc[i, 'X'] / (
                Vnom ** 2 * 1000 / (Snom)) for i in System_Data_Lines.index}
    Imax = {(System_Data_Lines.loc[i, 'FROM'], System_Data_Lines.loc[i, 'TO']): System_Data_Lines.loc[i, 'Imax'] / (
    (Snom / Vnom)) for i in System_Data_Lines.index}
    OT = [System_Time_n_Scale.loc[i, 'OT'] for i in System_Time_n_Scale.index]
    sc = {OT[i]: System_Time_n_Scale.loc[i, 'sc'] for i in System_Time_n_Scale.index}
    spv = {OT[i]: System_Time_n_Scale.loc[i, 'spv'] for i in System_Time_n_Scale.index}





    PDs = {(Ob[i], OT[t], Os[s]) : System_Time_n_Scale.loc[t, 'sc']*System_Data_Nodes.loc[i, 'PDn']/Snom + np.random.normal(0, System_Time_n_Scale.loc[t, 'sc']*System_Data_Nodes.loc[i, 'PDn']/Snom*cv) for i in System_Data_Nodes.index for t in System_Time_n_Scale.index for s in Os}
    SOCini = {(Ost[i], Os[s]) : np.random.uniform(System_energy_storage.loc[i, 'SOCm'], System_energy_storage.loc[i, 'SOCM'])/100 for i in System_energy_storage.index for s in Os}
    Ppv = {(Ob[i], OT[t], Os[s]) : Pct_penetration*System_Data_Nodes.loc[i, 'PVn'] / Snom + np.random.normal(0, Pct_penetration*System_Data_Nodes.loc[i, 'PVn'] / Snom*cv) for i in System_Data_Nodes.index for t in System_Time_n_Scale.index for s in Os}



    Data_Network = [Ob, Ol, Tb, PD, QD, R, X, Imax, OT, sc, Os, PDs, Ost, SOCini, SOCM, SOCm, PmaxE, EC, Cess, Ppv, spv]


    return Data_Network