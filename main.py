import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import fixedArms as fA
import generateArms as gA

if __name__ == '__main__':
    K_list = np.array([2])
    varyingK = True if len(K_list) > 1 else False
    T_list = np.arange(1, 21) * 500
    numArmDists = 1
    alpha = 0
    startSim = 0
    endSim = 250
    pw = 1 / 2

    # armInstances = gA.generateArms(K_list, T_list, numArmDists, alpha)
    armInstances = gA.generateMultipleArms(K_list, T_list, numArmDists, pw)

    start = time.time()
    naiveUCB1 = fA.naiveUCB1(armInstances, startSim, endSim, K_list, T_list, start)
    # ADAETC = fA.ADAETC(armInstances, startSim, endSim, K_list, T_list)
    # ETC = fA.ETC(armInstances, startSim, endSim, K_list, T_list)
    # NADAETC = fA.NADAETC(armInstances, startSim, endSim, K_list, T_list)
    # UCB1_stopping = fA.UCB1_stopping(armInstances, startSim, endSim, K_list, T_list)
    print("took " + str(time.time() - start) + " seconds; arms within " + str(pw) + " power of respective T values")
    print("Larger the ratio of difference between cumulative rewards to the total")
    print("cumulative rewards, better the performance of max objective")

    # df_ADAETC = pd.DataFrame({'T': T_list, 'Regret': ADAETC['regret'], 'Standard Error': ADAETC['standardError'],
    #                           'Ratio of pulls': ADAETC['pullRatios']})
    # df_ADAETC.to_csv('ADAETC.csv', index=False)
    # df_naiveUCB1 = pd.DataFrame({'T': T_list, 'Regret': naiveUCB1['regret'],
    #                              'Standard Error': naiveUCB1['standardError'],
    #                              'Ratio of pulls': naiveUCB1['pullRatios']})
    # for i in range(4):
    #     colName = 'Number of Switches - ' + str(i + 1)
    #     df_naiveUCB1[colName] = naiveUCB1['numSwitches'][i]
    # for i in range(4):
    #     colName = 'StErr of Switches - ' + str(i + 1)
    #     df_naiveUCB1[colName] = naiveUCB1['numSwitchErrors'][i]
    # df_naiveUCB1.to_csv('naiveUCB1.csv', index=False)

    # CHARTS
    # pull ratios
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.plot(T_list, naiveUCB1['pullRatios'])
    plt.title('The most pulled arm')
    plt.ylabel('average pulls spent across sims of ' + str(endSim))
    plt.xlabel('T')
    # plt.savefig('pullRatios.png')
    plt.show()

    # total switches
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.plot(T_list, sum(naiveUCB1['numSwitches']))
    plt.errorbar(T_list, sum(naiveUCB1['numSwitches']), yerr=sum(naiveUCB1['numSwitchErrors']), fmt='o')
    plt.title('The switches between arms')
    plt.ylabel('average number of switches across sims of ' + str(endSim))
    plt.xlabel('T')
    # plt.savefig('numberOfSwitches.png')
    plt.show()

    # quartered switches
    # for i in range(4):
    #     plt.plot(T_list, naiveUCB1['numSwitches'][i], label=i + 1)
    #     plt.errorbar(T_list, naiveUCB1['numSwitches'][i], yerr=naiveUCB1['numSwitchErrors'][i], fmt='o')
    # plt.legend(loc="upper left")
    # plt.title('The switches between arms')
    # plt.ylabel('average number of switches')
    # plt.xlabel('T')
    # # plt.savefig('numberOfSwitches_quartered.png')
    # plt.show()
