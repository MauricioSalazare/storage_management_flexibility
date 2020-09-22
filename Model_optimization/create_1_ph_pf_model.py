# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------

from pyomo.environ import *


def create_1_ph_pf_model(Vnom, Snom, Vmin, Vmax, Data_Network):
    # Data Processing
    Ob = Data_Network[0]
    Ol = Data_Network[1]
    Tb = Data_Network[2]
    PD = Data_Network[3]
    QD = Data_Network[4]
    R = Data_Network[5]
    X = Data_Network[6]
    Imax = Data_Network[7]
    OT = Data_Network[8]
    sc = Data_Network[9]
    Os = Data_Network[10]
    PDs = Data_Network[11]
    Ost = Data_Network[12]

    SOCini = Data_Network[13]
    SOCM = Data_Network[14]
    SOCm = Data_Network[15]
    PmaxE = Data_Network[16]
    EC = Data_Network[17]
    Cess = Data_Network[18]

    Ppv = Data_Network[19]
    spv = Data_Network[20]




    # Type of Model
    model = ConcreteModel()

    # Define Sets
    model.Ob = Set(initialize=Ob)
    model.Ol = Set(initialize=Ol)
    model.OT = Set(initialize=OT)
    model.Os = Set(initialize=Os)
    model.Ost = Set(initialize=Ost)

    # Define Parameters
    model.Vnom = Param(initialize=Vnom, mutable=True)
    model.Snom = Param(initialize=Snom, mutable=True)
    model.Vmin = Param(initialize=Vmin, mutable=True)
    model.Vmax = Param(initialize=Vmax, mutable=True)
    model.Tb = Param(model.Ob, initialize=Tb, mutable=True)
    model.PD = Param(model.Ob, initialize=PD, mutable=True)  # Node demand
    model.QD = Param(model.Ob, initialize=QD, mutable=True)  # Node demand
    model.PDs = Param(model.Ob, model.OT, model.Os, initialize=PDs, mutable=True)  # Node demand
    model.R = Param(model.Ol, initialize=R, mutable=True)  # Line resistance
    model.X = Param(model.Ol, initialize=X, mutable=True)  # Line resistance
    model.Imax = Param(model.Ol, initialize=Imax, mutable=True)  # Line resistance
    model.sc = Param(model.OT, initialize=sc, mutable=True)  # Line resistance

    # model.SOCini = Param(model.Ost, initialize=SOCini, mutable=True)  # Node demand
    model.SOCini = Param(model.Ost, model.Os, initialize=SOCini, mutable=True)  # Node demand
    model.SOCM = Param(model.Ost, initialize=SOCM, mutable=True)  # Node demand
    model.SOCm = Param(model.Ost, initialize=SOCm, mutable=True)  # Node demand
    model.EC = Param(model.Ost, initialize=EC, mutable=True)  # Node demand
    model.Cess = Param(model.Ost, initialize=Cess, mutable=True)  # Node demand
    model.PmaxE = Param(model.Ost, initialize=PmaxE, mutable=True)  # Node demand

    model.Ppv = Param(model.Ob, model.OT, model.Os, initialize=Ppv, mutable=True)  # Node demand
    model.spv = Param(model.OT, initialize=spv, mutable=True)  # Line resistance


    def R_init_rule(model, i, j):
        return (model.R[i, j])

    model.RM = Param(model.Ol, initialize=R_init_rule)  # Line resistance

    def X_init_rule(model, i, j):
        return (model.X[i, j])

    model.XM = Param(model.Ol, initialize=X_init_rule)  # Line resistance

    # def P_init_rule(model,i,t):
    #     return (model.PD[i]*model.sc[t])
    # model.PM = Param(model.Ob, model.OT, initialize= P_init_rule) # Line resistance
    #
    # def Q_init_rule(model, i,t):
    #     return (model.QD[i]*model.sc[t])
    # model.QM = Param(model.Ob, model.OT, initialize= Q_init_rule) # Line resistance

    def QDs_init_rule(model, i, t, s):
        if model.PD[i] == 0.0:
            return 0.0
        else:
            return model.PDs[i, t, s] * tan(atan(model.QD[i] / (model.PD[i])))

    model.QDs = Param(model.Ob, model.OT, model.Os, initialize=QDs_init_rule)  # Line resistance

    # Define Variables
    model.P = Var(model.Ol, model.OT, model.Os, initialize=0)  # Acive power flowing in lines
    model.Q = Var(model.Ol, model.OT, model.Os, initialize=0)  # Reacive power flowing in lines
    model.I = Var(model.Ol, model.OT, model.Os, initialize=0)  # Current of lines
    model.cf = Var(model.Ob, model.OT, initialize=0.0)
    model.SOC = Var(model.Ost, model.OT, model.Os, initialize=0.0)
    model.Pess = Var(model.Ost, model.OT, model.Os, initialize=0.0)

    def PS_init_rule(model, i, t, s):
        if model.Tb[i] == 0:
            temp = 0.0
            model.PS[i, t, s].fixed = True
        else:
            temp = 0.0
        return temp

    model.PS = Var(model.Ob, model.OT, model.Os, initialize=PS_init_rule)  # Active power of the SS

    def QS_init_rule(model, i, t, s):
        if model.Tb[i] == 0:
            temp = 0.0
            model.QS[i, t, s].fixed = True
        else:
            temp = 0.0
        return temp

    model.QS = Var(model.Ob, model.OT, model.Os, initialize=QS_init_rule)  # Active power of the SS

    # Voltafe of nodes
    def Voltage_init(model, i, t, s):
        if model.Tb[i] == 1:
            temp = 1.0  # model.Vnom
            model.V[i, t, s].fixed = True
        else:
            temp = 1.0  # model.Vnom
            model.V[i, t, s].fixed = False
        return temp

    model.V = Var(model.Ob, model.OT, model.Os, initialize=Voltage_init)

    # %% Define Objective Function
    def act_loss(model):
        return (1 / len(Os) * sum(
            sum(sum(model.RM[i, j] * (model.I[i, j, t, s]) for i, j in model.Ol) for t in model.OT) for s in model.Os))

    model.obj = Objective(rule=act_loss)

    def active_power_flow_rule(model, k, t, s):
        return (sum(model.P[j, i, t, s] for j, i in model.Ol if i == k) - sum(
            model.P[i, j, t, s] + model.RM[i, j] * (model.I[i, j, t, s]) for i, j in model.Ol if k == i) +
                model.PS[k, t, s] + sum(model.Pess[b,t,s] for b in model.Ost if b == k) + model.Ppv[k,t,s]*model.spv[t] == model.PDs[k, t, s])  #

    model.active_power_flow = Constraint(model.Ob, model.OT, model.Os, rule=active_power_flow_rule)

    def reactive_power_flow_rule(model, k, t, s):
        return (sum(model.Q[j, i, t, s] for j, i in model.Ol if i == k) - sum(
            model.Q[i, j, t, s] + model.XM[i, j] * (model.I[i, j, t, s]) for i, j in model.Ol if k == i) + model.QS[
                    k, t, s] == model.QDs[k, t, s])

    model.reactive_power_flow = Constraint(model.Ob, model.OT, model.Os, rule=reactive_power_flow_rule)

    def voltage_drop_rule(model, i, j, t, s):
        return (model.V[i, t, s] - 2 * (model.RM[i, j] * model.P[i, j, t, s] + model.XM[i, j] * model.Q[i, j, t, s]) - (
                    model.RM[i, j] ** 2 + model.XM[i, j] ** 2) * model.I[i, j, t, s] - model.V[j, t, s] == 0)

    model.voltage_drop = Constraint(model.Ol, model.OT, model.Os, rule=voltage_drop_rule)

    def define_current_rule(model, i, j, t, s):
        return ((model.I[i, j, t, s]) * (model.V[j, t, s]) == model.P[i, j, t, s] ** 2 + model.Q[i, j, t, s] ** 2)

    model.define_current = Constraint(model.Ol, model.OT, model.Os, rule=define_current_rule)

    # def current_limit_rule(model, i, j, t, s):
    #     return (model.I[i, j, t, s] <= Imax[i, j] ** 2)
    #
    # model.current_limit = Constraint(model.Ol, model.OT, model.Os, rule=current_limit_rule)

    def voltage_limit_rule(model, i, t, s):
        return (model.Vmin ** 2, model.V[i, t, s], model.Vmax ** 2)

    model.voltage_limit = Constraint(model.Ob, model.OT, model.Os, rule=voltage_limit_rule)




    ##############################################
    # Storage constraints
    ##############################################

    def storage_soc_rule(model, i,t,s):
        if t == 1:
            return (model.SOC[i, t, s] == model.SOCini[i,s] - ((24/len(model.OT)) / model.EC[i]) * (model.Pess[i, t, s]))
        else:
            return (model.SOC[i,t,s] == model.SOC[i, t - 1, s] - ((24/len(model.OT))/ model.EC[i]) * (model.Pess[i,t,s]))
    model.storage_soc = Constraint(model.Ost, model.OT, model.Os, rule=storage_soc_rule)
    #
    def storage_pess_rule(model, i,t,s):
        return (-model.PmaxE[i], model.Pess[i,t,s], model.PmaxE[i])

    model.storage_pess = Constraint(model.Ost, model.OT, model.Os, rule=storage_pess_rule)

    def storage_soc_rule2(model, i,t,s):
        return (model.SOCm[i], model.SOC[i,t,s], model.SOCM[i])
    model.storage_soc2 = Constraint(model.Ost, model.OT, model.Os, rule=storage_soc_rule2)

    def storage_soc_rule3(model, i,t,s):
        if t >= len(model.OT):
            return (model.SOCini[i, s] <= model.SOC[i, t, s])
        else:
            return (model.SOCm[i], model.SOC[i,t,s], model.SOCM[i])
    model.storage_soc3 = Constraint(model.Ost, model.OT, model.Os, rule=storage_soc_rule3)


    # # # #

    return model
