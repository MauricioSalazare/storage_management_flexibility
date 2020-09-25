# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------

#####################################################################################
#####################################################################################
import math

def print_results(result, n, Pct_penetration, MC_s):
    name_file = 'ems_optimization_{}_{}.csv'.format(Pct_penetration,MC_s)

    if n == 0:
        with open(name_file, 'w') as f:
            f.write("Scenario,time,")
            for i in result.Ob:
                f.write("v_%d," % i)
            f.write("Loading, SOC, storage_P, storage_Q")
            f.write("\n")

            for t in result.OT:
                # for s in result.Os:
                f.write("%d,%d," % (n, t))
                for i in result.Ob:
                    f.write("%.6f," % (math.sqrt(result.V[i, t, 0].value)))
                f.write("%.6f," % (math.sqrt(result.I[1, 2, t, 0].value)/result.Imax[1,2].value*100))
                for b in result.Ost:
                    f.write("%.6f, %.6f, %.6f" % (result.SOC[b, t, 0].value*100, result.Pess[b, t, 0].value*result.Snom.value, result.Qess[b, t, 0].value*result.Snom.value))
                f.write("\n")

        f.close()

    else:
        with open(name_file, 'a') as f:
            # f.write("\n")
            for t in result.OT:
                # for s in result.Os:
                f.write("%d,%d," % (n, t))
                for i in result.Ob:
                    f.write("%.6f," % (math.sqrt(result.V[i, t, 0].value)))
                f.write("%.6f," % (math.sqrt(result.I[1, 2, t, 0].value)/result.Imax[1,2].value*100))
                for b in result.Ost:
                    f.write("%.6f, %.6f, %.6f" % (result.SOC[b, t, 0].value*100, result.Pess[b, t, 0].value*result.Snom.value, result.Qess[b, t, 0].value*result.Snom.value))
                f.write("\n")
        f.close()

    A = 1

    return A
    #
