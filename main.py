import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import fixedArms as fA
import generateArms as gA
import fixedArmsRotting as fAR


def rotting(K_list_, T_list_, numArmDists_, alpha__, beta__, startSim_, endSim_, pw_):
    print("Running rotting bandits")
    start_ = time.time()
    armInstances_ = gA.generateRottingArms(K_list_[0], T_list_, numArmDists_, alpha__, beta__)
    print("Running rotting bandits")
    naiveUCB1_ = fAR.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_)
    ADAETC_ = fAR.ADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_)
    rotting_ = fAR.Rotting(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_,
                           sigma=0.25, deltaZero=2, alpha=0.05)
    print("took " + str(time.time() - start_) + " seconds; alpha " + str(alpha__) + ", beta " + str(beta__))

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'Rotting': rotting_}


def mEqOne(K_list_, T_list_, numArmDists_, startSim_, endSim_, alpha__, pw_):
    print("Running m = 1")
    start_ = time.time()
    armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__)
    # armInstances_ = gA.generateMultipleArms(K_list, T_list, numArmDists, pw_)
    # for i in range(10):
    #     armInstances_ = gA.generateTwoArms(T_list_, numArmDists_, delta=np.ones(5) * 0.05 * (i + 1))
    print("Running m=1")
    naiveUCB1_ = fA.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_, start_)
    ADAETC_ = fA.ADAETC(armInstances_, startSim_, endSim, K_list, T_list)
    ETC_ = fA.ETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
    NADAETC_ = fA.NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
    UCB1_stopping_ = fA.UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_)
    print("took " + str(time.time() - start_) + " seconds")

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'ETC': ETC_,
            'NADAETC': NADAETC_,
            'UCB1-s': UCB1_stopping_}


def mGeneral(K_list_, T_list_, numArmDists_, startSim_, endSim_, m_, alpha__, pw_):
    print("Running m =", m_)
    start_ = time.time()
    armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__)
    # armInstances_ = gA.generateMultipleArms(K_list_, T_list_, numArmDists_, pw_)
    print("Running m =", m_)
    RADAETC_ = fA.RADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, startSim_, endSim_, K_list, T_list_, m_)
    m_NADAETC_ = fA.m_NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    print("took " + str(time.time() - start_) + " seconds")

    return {'RADAETC': RADAETC_,
            'ADAETC': m_ADAETC_,
            'ETC': m_ETC_,
            'NADAETC': m_NADAETC_,
            'UCB1-s': m_UCB1_stopping_}


if __name__ == '__main__':
    K_list = np.array([2])
    varyingK = True if len(K_list) > 1 else False
    T_list = np.arange(1, 6) * 100
    m = 2
    numArmDists = 10
    alpha_ = 0.4  # can be used for both
    beta_ = 0.1  # for rotting bandits
    startSim = 0
    endSim = 10
    pw = 1 / 2  # used for both, larger pw means higher variance in mean changes for rotting
    # larger pw means closer mean rewards in the arm instances generated

    # rotting bandits part
    rotting(K_list, T_list, numArmDists, alpha_, beta_, startSim, endSim, pw)
    exit()

    # fixed mean rewards throughout, m = 1
    # mEqOne(K_list, T_list, numArmDists, startSim, endSim, alpha_, pw)
    # exit()

    # fixed mean rewards throughout, m > 1
    # mGeneral(K_list, T_list, numArmDists, startSim, endSim, m, alpha_, pw)
    # exit()

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
