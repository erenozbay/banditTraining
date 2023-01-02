import matplotlib.pyplot as plt
import pandas as pd
from cohortGenerator import *


def init_res():
    res_ = {'UCB1': {}, 'ADAETC': {}, 'ETC': {}, 'NADAETC': {}, 'UCB1-s': {}, 'SuccElim': {}}
    for i in res_.keys():
        res_[i]['Regret'], res_[i]['Reward'], res_[i]['cumrew'] = [], [], []
        res_[i]['standardError'], res_[i]['cumReg'] = [], []
    return res_


def store_res(res_, generateIns__, dif, inputDict_, key_):
    res_[key_]['regret_' + str(dif)] = inputDict_['regret']
    res_[key_]['Regret'].append(inputDict_['regret'][0])  # position zero is okay because these have a single T value
    res_[key_]['cumrew_' + str(dif)] = inputDict_['cumreward']
    res_[key_]['cumrew'].append(inputDict_['cumreward'][0])
    res_[key_]['cumReg'].append(inputDict_['cumReg'][0])
    res_[key_]['Reward'].append(inputDict_['reward'][0])
    if generateIns__ == 1:
        res_[key_]['standardError'].append(inputDict_['standardError_perSim'])
    else:
        res_[key_]['standardError'].append(inputDict_['standardError'][0])
    return res_


def plot_varying_delta(res_, delt_, numSim, T_, K_, generateIns__, alp, numOpt__, UCBin_=False, title='Regret'):
    bw = 0.15  # bar width
    naive_ucb1 = res_['UCB1']
    adaetc = res_['ADAETC']
    etc = res_['ETC']
    nadaetc = res_['NADAETC']
    ucb1s = res_['UCB1-s']
    succ_elim = res_['SuccElim']

    length = len(adaetc[title]) if title != 'cumReg' else len(adaetc[title]) - 1
    bar1 = np.arange(length) if title != 'cumReg' else np.arange(length) + 0.15
    bar2 = [x + bw for x in bar1]
    bar3 = [x + bw for x in bar2]
    bar4 = [x + bw for x in bar3]
    bar5 = [x + bw for x in bar4]
    bar6 = [x + bw for x in bar5]

    plt.figure(figsize=(12, 8), dpi=150)

    plt.bar(bar1, adaetc[title][-length:], yerr=adaetc['standardError'][-length:], color='r',
            width=bw, edgecolor='grey', label='ADA-ETC')
    plt.bar(bar2, etc[title][-length:], yerr=etc['standardError'][-length:], color='g',
            width=bw, edgecolor='grey', label='ETC')
    plt.bar(bar3, nadaetc[title][-length:], yerr=nadaetc['standardError'][-length:], color='magenta',
            width=bw, edgecolor='grey', label='NADA-ETC')
    plt.bar(bar4, ucb1s[title][-length:], yerr=ucb1s['standardError'][-length:], color='navy',
            width=bw, edgecolor='grey', label='UCB1-s')
    plt.bar(bar5, succ_elim[title][-length:], yerr=succ_elim['standardError'][-length:], color='purple',
            width=bw, edgecolor='grey', label='SuccElim - c(4)')
    if UCBin_:
        plt.bar(bar6, naive_ucb1[title][-length:], yerr=naive_ucb1['standardError'][-length:], color='b',
                width=bw, edgecolor='grey', label='UCB1')

    chartTitle = ''
    if title == 'cumrew':
        chartTitle = 'Cumulative Reward'
    elif title == 'cumReg':
        chartTitle = 'Sum Objective Regret'
    elif title == 'Reward':
        chartTitle = 'Best Arm Reward'
    elif title == 'Regret':
        chartTitle = 'Max Objective Regret'
    plt.ylabel(chartTitle, fontsize=15)
    plt.xlabel(r'$\Delta$', fontweight='bold', fontsize=15)
    plt.xticks([x + bw for x in bar1], delt_[-length:])

    plt.legend(loc="upper left")

    if K_ == 2:
        plt.savefig('res/' + str(K_) + 'arms_halfHalfDelta_' + str(numOpt__) + 'optArms_' + title + '_' + str(numSim) +
                    'sims_T' + str(T_) + '_' + str(generateIns__) + 'inst_UCB' + str(UCBin_) + '.eps',
                    format='eps', bbox_inches='tight')
    else:
        plt.savefig('res/' + str(K_) + 'arms_halfHalfDelta_' + str(numOpt__) + 'optArms_' + title + '_' + str(numSim) +
                    'sims_T' + str(T_) + '_' + str(generateIns__) + 'inst_' + str(alp) + 'alpha_UCB' +
                    str(UCBin_) + '.eps', format='eps', bbox_inches='tight')

    plt.cla()


def plot_fixed_m(i, K_list_, T_list, naiveUCB1_, TS, ADAETC_, ETC_,
                 NADAETC_, UCB1_stopping_, SuccElim_, Switching_, params_, BAI_ETC=None):
    numOpt_, alpha__, totSims_, Switch_do = params_['numOpt'], params_['alpha'], params_['totalSim'], params_['Switch']
    numArmDists_, constant_c, delt_, m_ = params_['numArmDists'], params_['c'], params_['delta'], params_['m']
    NADA = params_['NADA']

    if len(T_list) <= 10:
        plt.figure(figsize=(7, 5), dpi=100)
    elif len(T_list) <= 20:
        plt.figure(figsize=(14, 10), dpi=100)
    else:
        plt.figure(figsize=(28, 20), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    if i == 0:  # with UCB, m=1
        sizes = 6 if len(T_list) > 10 else 4
        plt.plot(T_list, naiveUCB1_['regret'], color='b', label='UCB1-I')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, TS['regret'], color='purple', label='TS')
        plt.errorbar(T_list, TS['regret'], yerr=TS['standardError'],
                     color='purple', fmt='o', markersize=sizes, capsize=sizes)
    if i < 2:  # without UCB, m=1
        sizes = 6 if len(T_list) > 10 else 4
        plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, ETC_['regret'], color='mediumseagreen', label='ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='mediumseagreen', fmt='o', markersize=sizes, capsize=sizes)
        if NADA == 'yes':
            plt.plot(T_list, NADAETC_['regret'], color='magenta', label='NADA-ETC')
            plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                         color='magenta', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='UCB1-I-s')
        plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=sizes, capsize=sizes)
        if SuccElim_ is not None:
            plt.plot(T_list, SuccElim_['regret'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
            plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                         color='purple', fmt='o', markersize=sizes, capsize=sizes)
    if i == 2:  # general m plots
        plt.plot(T_list, naiveUCB1_['regret'], color='b', label='m-UCB1-I')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['regret'], color='r', label='m-ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['regret'], color='mediumseagreen', label='m-ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='mediumseagreen', fmt='o', markersize=4, capsize=4)
        if NADA == 'yes':
            plt.plot(T_list, NADAETC_['regret'], color='magenta', label='m-NADA-ETC')
            plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                         color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='m-UCB1-I-s')
        plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        if SuccElim_ is not None:
            plt.plot(T_list, SuccElim_['regret'], color='purple', label='RADA-ETC')  # THIS IS RADA-ETC for this part
            plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                         color='purple', fmt='o', markersize=4, capsize=4)
    if i == 3:  # cumulative regret plots for m=1
        plt.plot(T_list, naiveUCB1_['cumReg'], color='b', label='UCB1-I')
        plt.errorbar(T_list, naiveUCB1_['cumReg'], yerr=naiveUCB1_['standardError_cumReg'],
                     color='b', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['cumReg'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['cumReg'], yerr=ADAETC_['standardError_cumReg'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['cumReg'], color='mediumseagreen', label='ETC')
        plt.errorbar(T_list, ETC_['cumReg'], yerr=ETC_['standardError_cumReg'],
                     color='mediumseagreen', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, TS['cumReg'], color='purple', label='TS')
        plt.errorbar(T_list, TS['cumReg'], yerr=TS['standardError_cumReg'],
                     color='purple', fmt='o', markersize=4, capsize=4)
        if NADA == 'yes':
            plt.plot(T_list, NADAETC_['cumReg'], color='magenta', label='NADA-ETC')
            plt.errorbar(T_list, NADAETC_['cumReg'], yerr=NADAETC_['standardError_cumReg'],
                         color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['cumReg'], color='navy', label='UCB1-I-s')
        plt.errorbar(T_list, UCB1_stopping_['cumReg'], yerr=UCB1_stopping_['standardError_cumReg'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        if SuccElim_ is not None:
            plt.plot(T_list, SuccElim_['cumReg'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
            plt.errorbar(T_list, SuccElim_['cumReg'], yerr=SuccElim_['standardError'],
                         color='purple', fmt='o', markersize=4, capsize=4)
    if i == 4:  # only for UCB1 and Switching bandits
        sizes = 10 if len(T_list) > 20 else 4
        plt.plot(T_list, naiveUCB1_['regret'], color='blue', label='UCB1-I')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='blue', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, TS['regret'], color='mediumseagreen', label='TS')
        plt.errorbar(T_list, TS['regret'], yerr=TS['standardError'],
                     color='mediumseagreen', fmt='o', markersize=sizes, capsize=sizes)
        if BAI_ETC is not None:
            plt.plot(T_list, BAI_ETC['regret'], color='navy', label='BAI-ETC')
            plt.errorbar(T_list, BAI_ETC['regret'], yerr=BAI_ETC['standardError'],
                         color='navy', fmt='o', markersize=sizes, capsize=sizes)
        if Switch_do == 'yes':
            plt.plot(T_list, Switching_['regret'], color='mediumseagreen', label='Switch')
            plt.errorbar(T_list, Switching_['regret'], yerr=Switching_['standardError'],
                         color='mediumseagreen', fmt='o', markersize=4, capsize=4)
    if i == 5:  # only for UCB1 and Switching bandits, cumulative regret plots
        sizes = 10 if len(T_list) > 20 else 4
        plt.plot(T_list, naiveUCB1_['cumReg'], color='blue', label='UCB1-I')
        plt.errorbar(T_list, naiveUCB1_['cumReg'], yerr=naiveUCB1_['standardError_cumReg'],
                     color='blue', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, ADAETC_['cumReg'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['cumReg'], yerr=ADAETC_['standardError_cumReg'],
                     color='r', fmt='o', markersize=sizes, capsize=sizes)
        plt.plot(T_list, TS['cumReg'], color='mediumseagreen', label='TS')
        plt.errorbar(T_list, TS['cumReg'], yerr=TS['standardError_cumReg'],
                     color='mediumseagreen', fmt='o', markersize=sizes, capsize=sizes)
        if BAI_ETC is not None:
            plt.plot(T_list, BAI_ETC['cumReg'], color='navy', label='BAI-ETC')
            plt.errorbar(T_list, BAI_ETC['cumReg'], yerr=BAI_ETC['standardError_cumReg'],
                         color='navy', fmt='o', markersize=sizes, capsize=sizes)
        if Switch_do == 'yes':
            plt.plot(T_list, Switching_['cumReg'], color='mediumseagreen', label='Switch')
            plt.errorbar(T_list, Switching_['cumReg'], yerr=Switching_['standardError_cumReg'],
                         color='mediumseagreen', fmt='o', markersize=4, capsize=4)

    fontSize = 20 if len(T_list) > 10 else 15
    plt.xticks(T_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Regret', fontsize=fontSize) if i != 3 and i != 5 else plt.ylabel('Regret', fontsize=fontSize)
    plt.xlabel('T', fontsize=fontSize)

    plt.legend(loc="upper left") if i < 4 else plt.legend(loc="upper left", prop={'size': 12})
    if i == 0:  # with UCB, m = 1
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')
    elif i < 2:  # without UCB, m = 1
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_noUCB_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 2:  # general m
        plt.savefig('res/mEquals' + str(m_) + '_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' +
                    str(alpha__) + '_sim' + str(totSims_) + '_armDist' + str(numArmDists_) +
                    '.eps', format='eps', bbox_inches='tight')
    elif i == 3:  # m = 1, sum objective
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sumObj_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 4:  # only UCB and switch
        plt.savefig('res/Thalf/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_UCBandSwitch_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 5:  # only UCB and switch, sum objective
        plt.savefig('res/Thalf/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sumObj_UCBandSwitch_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')

    plt.cla()



def plot_market_mBased(algs, m, K, T, results, totalCohorts, numSim, replicate, correction, exploreLess, normalized=False):
    colors = ['red', 'navy', 'mediumseagreen', 'magenta', 'purple', 'blue']
    labels = list(algs.keys())
    counter = 0
    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    for keys in algs.keys():
        plt.plot(np.arange(1, m + 1).astype('str'), results[keys]['reward'], color=colors[counter], label=labels[counter])
        plt.errorbar(np.arange(1, m + 1).astype('str'), results[keys]['reward'], yerr=results[keys]['stError'], color=colors[counter],
                     fmt='o', markersize=4, capsize=4)
        counter += 1
    plt.ylabel('Reward', fontsize=13)
    plt.xlabel('m', fontsize=13)
    title = 'marketSim/rewardGeneration' if not normalized else 'marketSim/rewardGeneration_NormalizedRewards'
    title += '_m' + str(m) + '_K' + str(K) + '_T' + str(T) + '_sim' + str(numSim) + \
            '_replication' + str(replicate) + '_' + str(totalCohorts) + 'cohorts' + '_correction' + str(correction)
    title += '_smallerTau.eps' if exploreLess else '.eps'
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1.02))
    # plt.ylim(top=11000)
    plt.savefig(title, format='eps', bbox_inches='tight')
    plt.cla()


def excel_market(algs, queuedJobs, usedJobs, rewardGenerated, queuedActiveCohorts, graduatedActiveCohorts,
                cohortComingsAndGoings, lastDeactivatedCohort, makesACohort, correction):

    for keys in algs.keys():
        pd.DataFrame(np.column_stack((np.transpose(queuedJobs[keys]), np.transpose(usedJobs[keys]),
                                      np.transpose(rewardGenerated[keys]), np.transpose(queuedActiveCohorts[keys]),
                                      np.transpose(graduatedActiveCohorts[keys]))),
                     columns=['In Queue Jobs', 'Used Jobs', 'Reward Generated', 'Active Cohorts',
                              'Graduated Cohorts']).to_csv("marketSim/timeDeps_cohortSize" + str(makesACohort) +
                                                           "_" + keys + "_correct" + str(correction) +
                                                           ".csv", index=False)
        cohortComingsAndGoings[keys][:, 3] = cohortComingsAndGoings[keys][:, 2] - \
                                             cohortComingsAndGoings[keys][:, 1] + 1
        pd.DataFrame(cohortComingsAndGoings[keys][:(lastDeactivatedCohort[keys] + 1)],
                     columns=['Index', 'Activated', 'Deactivated', 'Life']).\
            to_csv("marketSim/cohortMoves_cohortSize" + str(makesACohort) + "_" + keys + "_correct" +
                   str(correction) + ".csv", index=False)



def DynamicMarketSim(algs, m, K, T, m_cohort, totalCohorts, workerArrival, armsGenerated, exploreLess=False,
                     rewardGrouping=10, excludeZeros=False, cumulative=False, excels=True, replicate=1,
                     correction=False):
    print("Correction?", correction)
    # m_cohort is what I use to mean how many arms will come out of a cohort
    # e.g., if K = 20, m = 5, and m_cohort = 1, then every 4 arm will make a cohort and ultimate reward will be
    # all rewards averaged and divided by 5. If m_cohort = 1, ult. reward is all rewards averaged.
    totalPeriods = totalCohorts * T
    workerArrProb = (K / T)
    makesACohort = int(m_cohort * int(K / m))  # how many workers make a cohort
    if T != 200 and correction:
        print("Some manual fix needed for T and T_budget parameters. Exiting...")
        exit()
    if correction:
        T_budget = 33 * (m_cohort == 1) + 70 * (m_cohort == 2) + 108 * (m_cohort == 3) + \
                 144 * (m_cohort == 4) + 180 * (m_cohort == 5)
    else:
        T_budget = int(T * makesACohort / K)
    print(makesACohort, " makes a cohort.", end=" ")
    print("That is, T is ", T_budget)
    print("exploreLess?", exploreLess)
    print("Excluding zero reward periods?", excludeZeros)
    if not excludeZeros:
        print("Grouping ultimate rewards every " + str(rewardGrouping) + " periods.")
    rewardGrouping = int(rewardGrouping)

    # generate all arms
    numAllArms = int(workerArrProb * totalPeriods * 1.2)  # room for extra arms in case more than planned shows up
    maxCohorts = int(numAllArms / makesACohort)

    finalRewards, finalRewardsNormalized = {}, {}
    for keys in algs.keys():
        finalRewards[keys] = 0
        finalRewardsNormalized[keys] = 0

    for t in range(replicate):
        generateCohorts = 0  # index used to "generate" the next ready cohort
        numWorkersArrived = 0  # keeps track of the cumulated workers arrived
        queuedJobs, usedJobs, numJobs = {}, {}, {}
        rewardGenerated, rewardOfCohort, allCohorts = {}, {}, {}
        queuedActiveCohorts, graduatedActiveCohorts = {}, {}
        activeCohorts, toBeDeactivatedCohorts, lastDeactivatedCohort, cohortComingsAndGoings = {}, {}, {}, {}
        for keys in algs.keys():
            queuedJobs[keys] = np.zeros(totalPeriods + 1)  # records all the leftover jobs in a period
            usedJobs[keys] = np.zeros(totalPeriods + 1)  # records all the used jobs in a period
            numJobs[keys] = 0  # keeps track of the number of available jobs at the beginning of a period
            rewardGenerated[keys] = np.zeros(totalPeriods + 1)  # records the reward generated in each time period
            queuedActiveCohorts[keys] = np.zeros(totalPeriods + 1) # records all active cohort in the beginning of a period
            graduatedActiveCohorts[keys] = np.zeros(totalPeriods + 1)  # records cumulative # deactivated cohorts at the end of a period
            lastDeactivatedCohort[keys] = 0  # records the index of the last deactivated cohort
            rewardOfCohort[keys] = np.zeros(maxCohorts)  # keeps track of the reward due to each cohort
            activeCohorts[keys] = []  # list of all active cohorts, chronologically ordered
            toBeDeactivatedCohorts[keys] = []  # cohorts that will be deactivated at the beginning of the next period
            cohortComingsAndGoings[keys] = np.zeros((maxCohorts, 4))  # records when a cohort gets activated and deactivated

            # put all the arms into the cohorts
            # budget = int(T * makesACohort / K)
            allCohorts[keys] = [CohortGenerate(cohortNum=i, armList=armsGenerated[i, :], K=makesACohort, T=T_budget,
                                               m=m_cohort, alg=keys, exploreLess=exploreLess) for i in range(maxCohorts)]


        for i in range(totalPeriods):
            for keys in algs.keys():
                numJobs[keys] += 1
            numWorkersArrived += workerArrival[i]
            # check if enough workers have arrived for the next cohort
            nextCohort = True if numWorkersArrived == makesACohort else False
            # if so, generate and activate the next cohort
            if nextCohort:
                numWorkersArrived = 0
                for keys in algs.keys():
                    allCohorts[keys][generateCohorts].generated = True  # not using this, just there
                    activeCohorts[keys].append(generateCohorts)
                    cohortComingsAndGoings[keys][generateCohorts, :] = np.array([generateCohorts, i, 0, 0])  # 1st index 0
                generateCohorts += 1

            for keys in algs.keys():
                for j in range(len(toBeDeactivatedCohorts[keys])):
                    # remove a deactivated cohort, the oldest cohort should deactivate first
                    activeCohorts[keys].remove(toBeDeactivatedCohorts[keys][j])

                toBeDeactivatedCohorts[keys] = []

                queuedActiveCohorts[keys][i] = len(activeCohorts[keys])  # note the active cohort during period i
                graduatedActiveCohorts[keys][i] = max(graduatedActiveCohorts[keys])
                # do job assignments within active cohorts
                for j in range(len(activeCohorts[keys])):
                    if numJobs[keys] >= m_cohort:
                        inProgress = allCohorts[keys][activeCohorts[keys][j]].step(budget=m_cohort)
                        # rewardGenerated[keys][i] += inProgress['realtime_reward']
                        if inProgress['done']:
                            graduatedActiveCohorts[keys][i] += 1  # keep track of the cumulative deactivated/graduated cohorts
                            rewardOfCohort[keys][activeCohorts[keys][j]] = inProgress['final_reward']
                            rewardGenerated[keys][i] += inProgress['final_reward']
                            toBeDeactivatedCohorts[keys].append(activeCohorts[keys][j])  # graduated cohort will be deactivated
                            lastDeactivatedCohort[keys] = activeCohorts[keys][j]
                            cohortComingsAndGoings[keys][activeCohorts[keys][j], 2] = i
                        numJobs[keys] -= m_cohort  # deduct the number of jobs used
                        usedJobs[keys][i] += m_cohort
                    else:
                        break
                queuedJobs[keys][i] = numJobs[keys]  # number of jobs idly waiting for workers

        for keys in algs.keys():
            for j in range(len(toBeDeactivatedCohorts[keys])):
                # remove a deactivated cohort, the oldest cohort should deactivate first
                activeCohorts[keys].remove(toBeDeactivatedCohorts[keys][j])
            queuedActiveCohorts[keys][totalPeriods] = len(activeCohorts[keys])

        for keys in algs.keys():
            finalRewards[keys] += sum(rewardOfCohort[keys][:lastDeactivatedCohort[keys]]) / replicate
            finalRewardsNormalized[keys] += sum(rewardOfCohort[keys][:lastDeactivatedCohort[keys]]) / replicate / \
                                  lastDeactivatedCohort[keys] / m_cohort

    if excels:
        # print(len(queuedJobs['ADA-ETC']))
        # print(len(usedJobs['ADA-ETC']))
        # print(len(rewardGenerated['ADA-ETC']))
        # print(len(queuedActiveCohorts['ADA-ETC']))
        # print(len(graduatedActiveCohorts['ADA-ETC']))
        excel_market(algs, queuedJobs, usedJobs, rewardGenerated, queuedActiveCohorts, graduatedActiveCohorts,
                     cohortComingsAndGoings, lastDeactivatedCohort, makesACohort, correction)


    if replicate == 1:
        print("=" * 25)
        print("Total workers arrived ", end=" ")
        print(sum(workerArrival[:totalPeriods]))
        print("=" * 25)
        for keys in algs.keys():
            print("Algorithm", keys)
            print("Queued jobs (by time) ", end=" ")
            print(queuedJobs[keys][:-1].astype(int))
            print("=" * 25)
            print("Used jobs (by time) ", end=" ")
            print(usedJobs[keys][:-1])
            print("=" * 25)
            print("Active cohorts (by time) ", end=" ")
            print(queuedActiveCohorts[keys])
            print("=" * 25)
            print("Graduated cohorts (by time) ", end=" ")
            print(graduatedActiveCohorts[keys][:-1])
            print("=" * 25)
            print("Total cohorts graduated ", end=" ")
            print(lastDeactivatedCohort[keys])
            print("=" * 25)
            print()
        for keys in algs.keys():
            pd.set_option('display.max_columns', None)
            print("Cohort rewards (by cohort), last deactivated", lastDeactivatedCohort[keys], ", ", keys,  end=": ")
            print(sum(rewardOfCohort[keys][:lastDeactivatedCohort[keys]]))

    return {'rews': finalRewards, 'rewsNormalized': finalRewardsNormalized,
            'QAC': queuedActiveCohorts}