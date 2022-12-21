import numpy as np
import matplotlib.pyplot as plt
import fixedArms as fA
import generateArms as gA


def sim_small_mid_large_m(armMeansArray_, arrayK_, arrayT_, m_, ucbPart_, pullDiv, alg):
    output = {'reward': -1e8, 'regret': 1e8}
    if m_ == 1:
        if alg == 'ada' or alg == 'rada':
            output = fA.ADAETC(armMeansArray_, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'nada':
            output = fA.NADAETC(armMeansArray_, 1, arrayK_, arrayT_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.UCB1_stopping(armMeansArray_, 1, arrayK_, arrayT_, ucbPart_, verbose=False)
        elif alg == 'ucb1':
            output = fA.naiveUCB1(armMeansArray_, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'etc':
            output = fA.ETC(armMeansArray_, 1, arrayK_, arrayT_, verbose=False, pullDiv=pullDiv)
    else:
        if alg == 'ada':
            output = fA.m_ADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'rada':
            output = fA.RADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'nada':
            output = fA.m_NADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.m_UCB1_stopping(armMeansArray_, 1, arrayK_, arrayT_, m_, ucbPart_, verbose=False)
        elif alg == 'ucb1':
            output = fA.m_naiveUCB1(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'etc':
            output = fA.m_ETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False, pullDiv=pullDiv)
    reward = output['reward']
    regret = output['regret']
    return {'reward': reward, 'regret': regret}


def sample_K_T_streams(numStreams_, totalPeriods_, meanK_, meanT_, m_vals_, geom=False):
    K_list_stream, T_list_stream = {}, {}
    for st in range(numStreams_):
        K_list_, T_list_ = np.zeros(totalPeriods_), np.zeros(totalPeriods_)
        for s in range(totalPeriods_):
            while True:
                if geom:
                    sample_K = np.random.geometric(1 / meanK_, 1)
                else:
                    sample_K = int(np.random.poisson(meanK_, 1))
                if sample_K >= max(2, 2 * max(m_vals_) / totalPeriods_):
                    K_list_[s] = sample_K
                    break
            while True:
                if geom:
                    sample_T = np.random.geometric(1 / meanT_, 1)
                else:
                    sample_T = int(np.random.poisson(meanT_, 1))
                if sample_T > 5 * sample_K:
                    T_list_[s] = sample_T
                    break
        K_list_stream[str(st)], T_list_stream[str(st)] = K_list_, T_list_

    return {'K_list_stream': K_list_stream, 'T_list_stream': T_list_stream}


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


def plot_fixed_m(i, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_,
                 NADAETC_, UCB1_stopping_, SuccElim_, Switching_, params_):
    numOpt_, alpha__, totSims_, Switch_do = params_['numOpt'], params_['alpha'], params_['totalSim'], params_['Switch']
    numArmDists_, constant_c, delt_, m_ = params_['numArmDists'], params_['c'], params_['delta'], params_['m']

    if len(T_list) <= 10:
        plt.figure(figsize=(7, 5), dpi=100)
    else:
        plt.figure(figsize=(14, 10), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    if i == 0:  # with UCB, m=1
        plt.plot(T_list, naiveUCB1_['regret'], color='b', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
    if i < 2:  # without UCB, m=1
        plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['regret'], color='mediumseagreen', label='ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='mediumseagreen', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, NADAETC_['regret'], color='magenta', label='NADA-ETC')
        plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                     color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='UCB1-s')
        plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, SuccElim_['regret'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
        plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                     color='purple', fmt='o', markersize=4, capsize=4)
    if i == 2:  # general m plots
        plt.plot(T_list, naiveUCB1_['regret'], color='b', label='m-UCB1')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['regret'], color='r', label='m-ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['regret'], color='mediumseagreen', label='m-ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='mediumseagreen', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, NADAETC_['regret'], color='magenta', label='m-NADA-ETC')
        plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                     color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='m-UCB1-s')
        plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, SuccElim_['regret'], color='purple', label='RADA-ETC')  # THIS IS RADA-ETC for this part
        plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                     color='purple', fmt='o', markersize=4, capsize=4)
    if i == 3:  # cumulative regret plots for m=1
        plt.plot(T_list, naiveUCB1_['cumReg'], color='b', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['cumReg'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['cumReg'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['cumReg'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['cumReg'], color='mediumseagreen', label='ETC')
        plt.errorbar(T_list, ETC_['cumReg'], yerr=ETC_['standardError'],
                     color='mediumseagreen', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, NADAETC_['cumReg'], color='magenta', label='NADA-ETC')
        plt.errorbar(T_list, NADAETC_['cumReg'], yerr=NADAETC_['standardError'],
                     color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['cumReg'], color='navy', label='UCB1-s')
        plt.errorbar(T_list, UCB1_stopping_['cumReg'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, SuccElim_['cumReg'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
        plt.errorbar(T_list, SuccElim_['cumReg'], yerr=SuccElim_['standardError'],
                     color='purple', fmt='o', markersize=4, capsize=4)
    if i == 4:  # only for UCB1 and Switching bandits
        plt.plot(T_list, naiveUCB1_['regret'], color='blue', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='blue', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        if Switch_do == 'yes':
            plt.plot(T_list, Switching_['regret'], color='mediumseagreen', label='Switch')
            plt.errorbar(T_list, Switching_['regret'], yerr=Switching_['standardError'],
                         color='mediumseagreen', fmt='o', markersize=4, capsize=4)
    if i == 5:  # only for UCB1 and Switching bandits, cumulative regret plots
        plt.plot(T_list, naiveUCB1_['cumReg'], color='blue', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['cumReg'], yerr=naiveUCB1_['standardError_cumReg'],
                     color='blue', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['cumReg'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['cumReg'], yerr=ADAETC_['standardError_cumReg'],
                     color='r', fmt='o', markersize=4, capsize=4)
        if Switch_do == 'yes':
            plt.plot(T_list, Switching_['cumReg'], color='mediumseagreen', label='Switch')
            plt.errorbar(T_list, Switching_['cumReg'], yerr=Switching_['standardError_cumReg'],
                         color='mediumseagreen', fmt='o', markersize=4, capsize=4)

    plt.xticks(T_list)
    plt.ylabel('Regret', fontsize=15) if i != 3 and i != 5 else plt.ylabel('Cumulative Regret', fontsize=15)
    plt.xlabel('T', fontsize=15)

    plt.legend(loc="upper left") if i < 4 else plt.legend(loc="upper left", prop={'size': 15})
    if i == 0:  # with UCB, m = 1
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_c' + str(constant_c) +
                    '_delta' + str(delt_) + '.eps', format='eps', bbox_inches='tight')
    elif i < 2:  # without UCB, m = 1
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_noUCB_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_c' +
                    str(constant_c) + '_delta' + str(delt_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 2:  # general m
        plt.savefig('res/mEquals' + str(m_) + '_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' +
                    str(alpha__) + '_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_delta' + str(delt_) +
                    '.eps', format='eps', bbox_inches='tight')
    elif i == 3:  # m = 1, sum objective
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sumObj_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_c' +
                    str(constant_c) + '_delta' + str(delt_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 4:  # only UCB and switch
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_UCBandSwitch_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_c' +
                    str(constant_c) + '_delta' + str(delt_) + '.eps', format='eps', bbox_inches='tight')
    elif i == 5:  # only UCB and switch, sum objective
        plt.savefig('res/mEquals1_' + str(numOpt_) + 'optArms_K' + str(K_list_[0]) + '_alpha' + str(alpha__) +
                    '_sumObj_UCBandSwitch_sim' + str(totSims_) + '_armDist' + str(numArmDists_) + '_c' +
                    str(constant_c) + '_delta' + str(delt_) + '.eps', format='eps', bbox_inches='tight')

    plt.cla()


def plot_marketSim(K_, T_, m_vals_, rews, stdevs, params_):
    numOpt_, alpha__, bestRew = params_['numOpt'], params_['alpha'], params_['bestReward']
    numArmDist_, totSims_ = params_['numArmDists'], params_['totalSim']
    if numOpt_ == 0:
        numOpt_ = 'no'
    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    colors = ['red', 'purple', 'mediumseagreen', 'magenta', 'navy', 'blue']
    labels = ['ADA-ETC', 'RADA-ETC', 'ETC', 'NADA-ETC', 'UCB1-s', 'UCB1']

    counter = 0
    for keys in rews.keys():
        plt.plot(m_vals_.astype('str'), rews[keys], color=colors[counter], label=labels[counter])
        plt.errorbar(m_vals_.astype('str'), rews[keys], yerr=stdevs[keys], color=colors[counter],
                     fmt='o', markersize=4, capsize=4)
        counter += 1
    plt.plot(m_vals_.astype('str'), bestRew, color='darkgreen', linestyle='--', label='Best')

    plt.ylabel('Reward', fontsize=13)
    plt.xlabel('m', fontsize=13)

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1.02))
    plt.savefig('res/marketSim_' + str(numOpt_) + 'OptArms_meanK' + str(K_) + '_meanT' + str(T_) + '_alpha' +
                str(alpha__) + '_sim' + str(totSims_) + '_armDist' + str(numArmDist_) + '.eps',
                format='eps', bbox_inches='tight')

    plt.cla()


class CohortGenerator:
    def __init__(self, cohortNum, armList, K, T, m):
        self.cohortNum = cohortNum
        self.generated = False
        self.K = K
        self.T = T
        self.m = m
        self.arms = armList
        self.pulls = np.zeros(self.K)
        self.indexhigh = np.zeros(self.K)
        self.indexlow = np.zeros(self.K)
        self.empirical_mean = np.zeros(self.K)
        self.cumulative_reward = np.zeros(self.K)
        self.pullEach = int(np.ceil(np.power(T / (K - m), 2 / 3))) if m > 1 else int(np.ceil(np.power(T / K, 2 / 3)))
        self.exploitPhase = False
        self.stopped = 0
        self.pullset = np.zeros(m)

    def Exploit(self):
        lcb_set = np.argsort(-self.indexlow)
        lcb = lcb_set[self.m - 1]
        indexhigh_copy = self.indexhigh.copy()
        indexhigh_copy[lcb_set[0:self.m]] = -1  # make sure that correct arms are excluded from max UCB step
        ucb = np.argsort(-indexhigh_copy)[0]
        pullset = lcb_set[0:self.m]
        if (self.indexlow[lcb] > self.indexhigh[ucb]) or ((self.indexlow[lcb] >= self.indexhigh[ucb])
                                                          and sum(self.pulls) > self.K):
            self.stopped = int(sum(self.pulls) / self.m)
            self.exploitPhase = True
            self.pullset = pullset

    def ADAETC(self, budget):
        done = False
        if not self.exploitPhase:  # do not check if I am in the exploitation phase after I get in it
            if any(self.pulls < 1):
                for i in range(self.K):
                    if self.pulls[i] == 0:
                        budget -= 1
                        pull = i
                        rew = np.random.binomial(1, self.arms[pull], 1)
                        self.cumulative_reward[pull] += rew
                        self.pulls[pull] += 1
                        self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                        up = 2 * np.sqrt(
                            max(np.log(self.T / (self.K * np.power(self.pulls[pull], 3 / 2))), 0) / self.pulls[pull])
                        self.indexhigh[pull] = self.empirical_mean[pull] + up * (self.pullEach > self.pulls[pull])
                        self.indexlow[pull] = self.empirical_mean[pull] - self.empirical_mean[pull] * (
                                    self.pullEach > self.pulls[pull])
                    if budget <= 0:
                        break
            if budget >= self.m:  # I want to pull m arms at all times, not anything less
                pullset = np.argsort(-self.indexhigh)
                for b in range(self.m):
                    pull = pullset[b]
                    rew = np.random.binomial(1, self.arms[pull], 1)
                    self.cumulative_reward[pull] += rew
                    self.pulls[pull] += 1
                    self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                    up = 2 * np.sqrt(
                        max(np.log(self.T / (self.K * np.power(self.pulls[pull], 3 / 2))), 0) / self.pulls[pull])
                    self.indexhigh[pull] = self.empirical_mean[pull] + up * (self.pullEach > self.pulls[pull])
                    self.indexlow[pull] = self.empirical_mean[pull] - self.empirical_mean[pull] * (
                                self.pullEach > self.pulls[pull])
                self.Exploit()   # check if I can exploit the next time

        else:  # already in the exploitation phase
            for b in range(self.m):
                pull = self.pullset[b]
                rew = np.random.binomial(1, self.arms[pull], 1)
                self.cumulative_reward[pull] += rew
                self.pulls[pull] += 1
                self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                up = 2 * np.sqrt(
                    max(np.log(self.T / (self.K * np.power(self.pulls[pull], 3 / 2))), 0) / self.pulls[pull])
                self.indexhigh[pull] = self.empirical_mean[pull] + up * (self.pullEach > self.pulls[pull])
                self.indexlow[pull] = self.empirical_mean[pull] - self.empirical_mean[pull] * (
                        self.pullEach > self.pulls[pull])
            done = True if (self.T - sum(self.pulls) < self.m) else False

        return {'budget': budget, # this budget I can use to return to the pool of jobs, maybe?
                'done': done,
                'reward': np.mean(self.cumulative_reward[self.pullset])}



def DynamicMarketSim(m, K, T, m_cohort, totalCohorts):
    # m_cohort is what I use to mean how many arms will come out of a cohort
    # e.g., if K = 20, m = 5, and m_cohort = 1, then every 4 arm will make a cohort and ultimate reward will be
    # all rewards averaged and divided by 5. If m_cohort = 1, ult. reward is all rewards averaged.
    totalPeriods = totalCohorts * T
    workerArrProb = (K / T)
    makesACohort = int(m_cohort * int(K / m))  # how many workers make a cohort
    generateCohorts = 0  # used to "generate" the next ready cohort
    queuedActiveCohorts = np.zeros(totalPeriods)  # records all active cohort in the current period
    graduatedActiveCohorts = np.zeros(totalPeriods)  # records all to be deactivated cohorts at the end of this period
    queuedJobs = np.zeros(totalPeriods)  # records all the jobs in a period

    # arrivals stored here
    numJobs = 0  # keeps track of the number of available jobs at the beginning of a period
    numWorkersArrived = 0  # keeps track of the cumulated workers arrived
    workerArrival = np.random.binomial(1, (np.ones(totalPeriods) * workerArrProb))  # worker arrival stream

    # generate all arms
    numAllArms = workerArrProb * np.power(totalPeriods, 1.5)  # room for extra arms in case more than planned shows up
    maxCohorts = int(numAllArms / makesACohort)
    rewardOfCohort = np.zeros(maxCohorts)  # keeps track of the reward due to each cohort
    activeCohorts = []  # list of all active cohorts, chronologically ordered
    toBeDeactivatedCohorts = []  # cohorts that will be deactivated at the beginning of the next period

    # generate the arms, single row contains the arms that will be in a single cohort
    armsGenerated = gA.generateArms(K_list=makesACohort, T_list=1, numArmDists=maxCohorts, alpha=0)

    # put all the arms into the cohorts
    allCohorts = [CohortGenerator(cohortNum=i, armList=armsGenerated[i, :], K=makesACohort,
                                  T=int(T * makesACohort / K), m=m_cohort) for i in range(maxCohorts)]

    for i in range(totalPeriods):
        for j in range(len(toBeDeactivatedCohorts)):
            # remove a deactivated cohort, the oldest cohort should deactivate first
            activeCohorts.remove(toBeDeactivatedCohorts[j])
        numJobs += 1
        numWorkersArrived += workerArrival[i]
        # check if enough workers have arrived for the next cohort
        nextCohort = True if numWorkersArrived % makesACohort == 0 else False
        # if so, generate and activate the next cohort
        if nextCohort:
            allCohorts[generateCohorts].generated = True  # not using this, just there
            activeCohorts.append(generateCohorts)
            generateCohorts += 1

        queuedActiveCohorts[i] = len(activeCohorts)
        # do job assignments within active cohorts
        for j in range(len(activeCohorts)):
            if numJobs >= m_cohort:
                inProgress = allCohorts[activeCohorts[j]].ADAETC(budget=m_cohort)
                if inProgress['done']:
                    graduatedActiveCohorts[i] += 1  # keep track of the deactivated/graduated cohort
                    rewardOfCohort[j] = inProgress['reward']
                numJobs -= m_cohort

        queuedJobs[i] = numJobs  # number of jobs waiting for workers

    print("Queued jobs (by time) ", end=" ")
    print(queuedJobs)
    print("=" * 25)
    print("Cohort rewards (by cohort) ", end=" ")
    print(rewardOfCohort)
    print("=" * 25)
    print("Active cohorts (by time) ", end=" ")
    print(queuedActiveCohorts)
    print("=" * 25)
    print("Total workers arrived ", end=" ")
    print(numWorkersArrived)
    print("=" * 25)
    print("Graduated cohorts (by time) ", end=" ")
    print(graduatedActiveCohorts)

