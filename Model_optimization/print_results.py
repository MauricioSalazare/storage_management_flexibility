# ------------------------------------------------------------------------
# Fair Load Profile Python Code developed by
# Juan S. Giraldo, TU Eindhoven, j.s.giraldo.chavarriaga@tue.nl
# ------------------------------------------------------------------------

#####################################################################################
#####################################################################################
import math


def print_results(model):
    dash1 = '-' * 100
    dash2 = '-' * 30
    print('\n')
    print(dash1)
    print("\t\t SUMMARY")
    print(dash1)
    print('\n')

    print('\n')
    print(dash2)
    print("\t\t VOLTAGE MAGNITUDES")
    print(dash2)
    for i in model.Ob:
        print('{:<8d}'.format(i), end=" ")
        for t in model.OT:
            print('{:>16.6f}'.format(math.sqrt(model.V[i, t].value)), end=" ")
        print('')
        #

    print('\n')
    print('\n')
    print(dash2)
    print("\t\t CURRENT MAGNITUDES")
    print(dash2)

    for (i, j) in model.Ol:
        print('{:>5d}{:^2s}{:<5d}'.format(i, '--', j), end=" ")
        for t in model.OT:
            print('{:>16.6f}'.format(math.sqrt(model.I[i, j, t].value) * model.Snom.value / model.Vnom.value), end=" ")
        print('')
        #

    print('\n')
    print('\n')
    print(dash2)
    print("\t\t FLEXIBILITY FACTORS")
    print(dash2)
    A = 0
    # for i in model.Ob:
    #     print('{:<8d}'.format(i), end=" ")
    #     for t in model.OT:
    #         print('{:>16.6f}'.format(model.K[i, t].value), end=" ")
    #         A = A + model.cf[i, t].value
    #     print('')

    print('\n')
    print('\n')
    print(dash2)
    print("\t\t TOTAL COST OF FLEXIBILITY")
    print(dash2)
    print('$ {:>1.4f}'.format(A*model.Snom.value), end=" ")

    return A
    #
