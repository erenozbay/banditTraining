import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import fixedArms as fA
import generateArms as gA
import fixedArmsRotting as fAR

if __name__ == '__main__':
    K_list = np.array([4])
    varyingK = True if len(K_list) > 1 else False
    T_list = np.arange(1, 6) * 100
    numArmDists = 250
    alpha_ = 0  # can be used for both
    beta_ = 0.01  # for rotting bandits
    startSim = 0
    endSim = 100
    pw = 1  # used for both, larger pw means higher variance in mean changes for rotting
            # larger pw means closer mean rewards in the arm instances generated

    # below items should be made neater, it's hard to see what is running

    # rotting bandits part
    # start = time.time()
    # armInstances = gA.generateRottingArms(K_list[0], T_list, numArmDists, alpha_, beta_)
    # naiveUCB1 = fAR.naiveUCB1(armInstances, startSim, endSim, K_list, T_list, pw)
    # ADAETC = fAR.ADAETC(armInstances, startSim, endSim, K_list, T_list, pw)
    # rotting = fAR.Rotting(armInstances, startSim, endSim, K_list, T_list, pw, sigma=0.25, deltaZero=2, alpha=0.05)
    # print("took " + str(time.time() - start) + " seconds; alpha " + str(alpha_) + ", beta " + str(beta_))

    # fixed mean rewards throughout, m = 1
    # start = time.time()
    # armInstances = gA.generateArms(K_list, T_list, numArmDists, alpha_)
    # armInstances = gA.generateMultipleArms(K_list, T_list, numArmDists, pw)
    # for i in range(10):
    #     armInstances = gA.generateTwoArms(T_list, numArmDists, delta=np.ones(5)*0.05*(i + 1))
    #     naiveUCB1 = fA.naiveUCB1(armInstances, startSim, endSim, K_list, T_list, start)
    #     ADAETC = fA.ADAETC(armInstances, startSim, endSim, K_list, T_list)
    #     ETC = fA.ETC(armInstances, startSim, endSim, K_list, T_list)
    #     NADAETC = fA.NADAETC(armInstances, startSim, endSim, K_list, T_list)
    #     UCB1_stopping = fA.UCB1_stopping(armInstances, startSim, endSim, K_list, T_list)
    #     print("took " + str(time.time() - start) + " seconds; arms within " + str(pw) + " power of respective T values")

    # fixed mean rewards throughout, m > 1
    start = time.time()
    armInstances = gA.generateArms(K_list, T_list, numArmDists, alpha_)
    # armInstances = gA.generateMultipleArms(K_list, T_list, numArmDists, pw)
    RADAETC = fA.RADAETC(armInstances, startSim, endSim, K_list, T_list, m=2)
    m_ADAETC = fA.m_ADAETC(armInstances, startSim, endSim, K_list, T_list, m=2)
    m_ETC = fA.m_ETC(armInstances, startSim, endSim, K_list, T_list, m=2)
    m_NADAETC = fA.m_NADAETC(armInstances, startSim, endSim, K_list, T_list, m=2)
    m_UCB1_stopping = fA.m_UCB1_stopping(armInstances, startSim, endSim, K_list, T_list, m=2)
    print("took " + str(time.time() - start) + " seconds")

    # DataFrames
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
    # plt.rc('axes', axisbelow=True)
    # plt.grid()
    # plt.plot(T_list, naiveUCB1['pullRatios'])
    # plt.title('The most pulled arm')
    # plt.ylabel('average pulls spent across sims of ' + str(endSim))
    # plt.xlabel('T')
    # plt.savefig('res/pullRatios.png')
    # # plt.show()
    # plt.cla()

    # total switches
    # plt.rc('axes', axisbelow=True)
    # plt.grid()
    # plt.plot(T_list, sum(naiveUCB1['numSwitches']))
    # plt.errorbar(T_list, sum(naiveUCB1['numSwitches']), yerr=sum(naiveUCB1['numSwitchErrors']), fmt='o')
    # plt.ylim(ymin=0)
    # plt.title('The switches between arms')
    # plt.ylabel('average number of switches across sims of ' + str(endSim))
    # plt.xlabel('T')
    # plt.savefig('res/numberOfSwitches.png')
    # plt.show()
    # plt.cla()

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
    # plt.cla()
