import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import fixedArms as fA
import generateArms as gA
import fixedArmsRotting as fAR
from scipy.stats import nbinom


# totalPeriods_ should be divisible by all values in m_vals, e.g., 12 periods, m_vals = [1, 2, 3, 6]
def marketSim(meanK_, meanT_, numArmDists_, numStreams_, totalPeriods_, m_vals_,
              alpha__, startSim_, endSim_, oneOptPerPeriod=False):

    # calls ADA-ETC or m-ADA=ETC depending on the m_ value
    def sim_small_mid_large_m(armMeansArray_, arrayK_, arrayT_, m_):
        if m_ == 1:
            ADAETC_ = fA.ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
            # ADAETC_ = fA.m_ADAETC(armMeansArray_, 0, 1, arrayK_,
            #                       arrayT_, 1, verbose=False)  # this shortens the exploration for m = 1 case
            reward = ADAETC_['reward']
            regret = ADAETC_['regret']
        else:
            m_ADAETC_ = fA.m_ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
            reward = m_ADAETC_['reward']
            regret = m_ADAETC_['regret']
        return {'reward': reward,
                'regret': regret}

    start_ = time.time()
    res = {}
    resreg = {}
    stdev = {}
    res['best'] = 0

    # a single instance will run with multiple simulations
    # a single instance will have a fixed list of K and T for each period as well as arm means
    # with each instance, ADA-ETC or m-ADA-ETC will be called
    # the ultimate average reward will be calculated based on what these models return and will be averaged over
    # simulations for that instance
    # no need for a global instance generation, the sequence can follow as
    # depending on the number of instances to create, create one instance and following that
    # run however many simulations you need to run and report an average 'average reward' out of this step
    # then the final analysis will be made over averaging these instances

    # fix numStreams_ many K and T streams, pay attention to the least number of arms required per period
    K_list_stream = {}
    T_list_stream = {}
    for st in range(numStreams_):
        res['stream_' + str(st)] = {}
        resreg['stream_' + str(st)] = {}
        stdev['stream_' + str(st)] = {}
        res['stream_' + str(st)]['best'] = 0
        K_list_ = np.zeros(totalPeriods_)
        T_list_ = np.zeros(totalPeriods_)
        for s in range(totalPeriods_):
            sampling_K = True
            sample_K = 1e4
            sampling_T = True
            while sampling_K:
                sample_K = int(np.random.poisson(meanK_, 1))
                # sample_K = np.random.geometric(1 / meanK_, 1)
                if sample_K >= max(2, 2 * max(m_vals_) / totalPeriods_):
                    K_list_[s] = sample_K
                    sampling_K = False
            while sampling_T:
                sample_T = int(np.random.poisson(meanT_, 1))
                # sample_T = np.random.geometric(1 / meanT_, 1)
                if sample_T > 5 * sample_K:
                    T_list_[s] = sample_T
                    sampling_T = False
        K_list_stream[str(st)] = K_list_
        T_list_stream[str(st)] = T_list_

    # run the simulation for each (K, T)-pair
    # fix (K, T) and following it, fix an instance with the correct number of arms, following the period-based K values
    # here we have the option of (1) either generation one optimal arm per period (by keeping oneOptPerPeriod = True)
    # which generates all the required arms across all periods, selects the best totalPeriods_-many of them and places
    # one of each to every period, and the rest of the arms required for the periods are placed randomly; or
    # (2) generate everything randomly
    for st in range(numStreams_):
        K_list_ = np.array(K_list_stream[str(st)])
        T_list_ = np.array(T_list_stream[str(st)])
        for a in range(numArmDists_):
            if (a + 1) % 10 == 0:
                print("Arm dist number ", str(a + 1), ", KnT stream number", str(st + 1), ", time ",
                      str(time.time() - start_), " from start.")

            # generate arm means, either each period has one optimal arm (across all periods) or no such arrangement
            armInstances_ = {}
            if oneOptPerPeriod:
                allArmInstances_ = gA.generateArms(K_list=np.array([sum(K_list_)]), T_list=np.array([sum(T_list_)]),
                                                   numArmDists=1, alpha=alpha__, verbose=False)
                # get the top totalPeriods_-many arms, put them aside in top_m and shuffle the remaining arms
                allArmInstances_ = np.sort(allArmInstances_[0])
                top_m = allArmInstances_[-totalPeriods_:]
                allArmInstances_ = allArmInstances_[:(int(sum(K_list_)) - totalPeriods_)]
                np.random.shuffle(allArmInstances_)
                col_s = 0
                for p in range(totalPeriods_):
                    armInstances_[str(p)] = np.concatenate((top_m[p],
                                                            allArmInstances_[col_s:(col_s + int(K_list_[p]) - 1)]),
                                                           axis=None)
                    col_s += int(K_list_[p]) - 1
            else:
                col_s = 0
                allArmInstances_ = np.zeros(int(sum(K_list_)))
                for p in range(totalPeriods_):
                    armInstances_[str(p)] = gA.generateArms(K_list=np.array([K_list_[p]]),
                                                            T_list=np.array([T_list_[p]]), numArmDists=1,
                                                            alpha=alpha__, verbose=False)
                    allArmInstances_[col_s:(col_s + int(K_list_[p]))] = np.array(armInstances_[str(p)])
                    col_s += int(K_list_[p])
                allArmInstances_ = np.sort(allArmInstances_)
                top_m = allArmInstances_[-totalPeriods_:]

            # get the average reward of the top m arms in an instance, in this case m = totalPeriods_
            best_top_m_avg_reward = np.mean(top_m)
            total_T = sum(T_list_) / totalPeriods_

            # run multiple simulations for varying values of m
            # if m = 1, every period is run individually and ADA-ETC is called using corresponding (K, T) pair
            # if m > 1, then multiple periods are combined and m-ADA-ETC is called using corresponding (K, T) pair which
            # for this case is summation of different K's and T's.
            for i in range(len(m_vals_)):
                # decide on the number of calls to simulation module and the corresponding m value
                m_val = m_vals_[i]  # say this is 3
                res['stream_' + str(st)]['m = ' + str(m_val)] = 0
                resreg['stream_' + str(st)]['reg: m = ' + str(m_val)] = 0
                calls = int(totalPeriods_ / m_val)  # and this is 6 / 3 = 2, i.e., totalPeriods_ = 6
                stdev_local = []
                for p in range(calls):  # so this will be 0, 1
                    indices = np.zeros(m_val)  # should look like 0, 1, 2;        3,  4,  5
                    for j in range(m_val):
                        indices[j] = p * m_val + j  # so this does the below example
                        # mval = 3; p = 0, j = 0 => 0; p = 0, j = 1 => 1; p = 0, j = 2 => 2
                        # mval = 3; p = 1, j = 0 => 3; p = 1, j = 1 => 4; p = 1, j = 2 => 5
                    # properly combine K and T values if needed
                    K_vals_ = 0
                    T_vals_ = 0
                    for j in range(m_val):
                        K_vals_ += int(K_list_[int(indices[j])])
                        T_vals_ += int(T_list_[int(indices[j])])

                    # depending on the K values, extract the correct set of arms
                    arms_ = np.zeros((1, int(K_vals_)))
                    col_start = 0
                    for j in range(m_val):
                        col_end = int(col_start + K_list_[int(indices[j])])
                        arms_[0, col_start:col_end] = armInstances_[str(int(indices[j]))]
                        col_start += int(K_list_[int(indices[j])])
                    # run the multiple simulations
                    for t in range(endSim_ - startSim_):
                        run = sim_small_mid_large_m(arms_, np.array([K_vals_]), np.array([T_vals_]), m_val)
                        rew = run['reward'].item() / (calls * int(endSim_ - startSim_))
                        reg = run['regret'].item() / (calls * int(endSim_ - startSim_))
                        # .item() is used to get the value, not as array
                        stdev_local.append(rew)
                        res['stream_' + str(st)]['m = ' + str(m_val)] += rew
                        resreg['stream_' + str(st)]['reg: m = ' + str(m_val)] += reg
                    stdev['stream_' + str(st)]['m = ' + str(m_val)] = np.sqrt(np.var(np.array(stdev_local)) /
                                                                              int(endSim_ - startSim_))

            res['stream_' + str(st)]['best'] += total_T * best_top_m_avg_reward / numArmDists_
    rewards = {}
    stdevs = {}
    regrets = {}
    best_rewards = 0
    once = True
    for j in range(len(m_vals_)):
        m_val = m_vals_[j]
        rewards['m = ' + str(m_val)] = 0
        stdevs['m = ' + str(m_val)] = 0
        regrets['reg: m = ' + str(m_val)] = 0
        for st in range(numStreams_):
            rewards['m = ' + str(m_vals_[j])] += res['stream_' + str(st)]['m = ' + str(m_val)] / numStreams_
            stdevs['m = ' + str(m_vals_[j])] += stdev['stream_' + str(st)]['m = ' + str(m_val)] / numStreams_
            regrets['reg: m = ' + str(m_vals_[j])] += resreg['stream_' + str(st)]['reg: m = ' + str(m_val)] / numStreams_
            if once:
                best_rewards += res['stream_' + str(st)]['best']
                once = False

    print()
    print("numArmDist", numArmDists_, "; alpha", alpha__, "; sims", endSim_ - startSim_,
          "; meanK", meanK_, "; meanT", meanT_, "; numStreams", numStreams_)
    print("One optimal arm per period?", oneOptPerPeriod, "; total periods", totalPeriods_)
    print('Rewards:')
    print('Best: ', best_rewards)
    for i in range(len(m_vals_)):
        print("m =", str(m_vals_[i]), ": ", round(rewards['m = ' + str(m_vals_[i])], 5))
    print()
    print('Standard deviation of rewards')
    for i in range(len(m_vals_)):
        print("m =", str(m_vals_[i]), ": ", round(stdevs['m = ' + str(m_vals_[i])], 5))
    print()
    print('Regrets')
    for i in range(len(m_vals_)):
        print("m =", str(m_vals_[i]), ": ", round(regrets['reg: m = ' + str(m_vals_[i])], 5))
    print("Done after ", str(round(time.time() - start_, 2)), " seconds from start.")
    return {'result': res,
            'regret': resreg,
            'stDev': stdev}


def rotting(K_list_, T_list_, numArmDists_, alpha__, beta__, startSim_, endSim_, pw_):
    print("Running rotting bandits")
    start_ = time.time()
    armInstances_ = gA.generateRottingArms(K_list_[0], T_list_, numArmDists_, alpha__, beta__)
    print("Running rotting bandits")
    naiveUCB1_ = fAR.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_)
    ADAETC_ = fAR.ADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_)
    rotting_ = fAR.Rotting(armInstances_, startSim_, endSim_, K_list_, T_list_, pw_,
                           sigma=1, deltaZero=2, alpha=0.1)
    print("took " + str(time.time() - start_) + " seconds; alpha " + str(alpha__) + ", beta " + str(beta__))

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'Rotting': rotting_}


def mEqOne_2arms(K_list_, T_list_, startSim_, endSim_, rng=11):
    def init_res():
        res_ = {'UCB1': {}, 'ADAETC': {}, 'ETC': {}, 'NADAETC': {}, 'UCB1-s': {}, 'SuccElim': {}}

        res_['UCB1']['Regret'] = []
        res_['ADAETC']['Regret'] = []
        res_['ETC']['Regret'] = []
        res_['NADAETC']['Regret'] = []
        res_['UCB1-s']['Regret'] = []
        res_['SuccElim']['Regret'] = []

        res_['UCB1']['Reward'] = []
        res_['ADAETC']['Reward'] = []
        res_['ETC']['Reward'] = []
        res_['NADAETC']['Reward'] = []
        res_['UCB1-s']['Reward'] = []
        res_['SuccElim']['Reward'] = []

        res_['UCB1']['cumrew'] = []
        res_['ADAETC']['cumrew'] = []
        res_['ETC']['cumrew'] = []
        res_['NADAETC']['cumrew'] = []
        res_['UCB1-s']['cumrew'] = []
        res_['SuccElim']['cumrew'] = []
        return res_

    def store_res(res_, dif, naiveUCB1__, ADAETC__, ETC__, NADAETC__, UCB1_stopping__, SuccElim__):
        res_['UCB1']['regret_' + str(dif)] = naiveUCB1__['regret']
        res_['UCB1']['Regret'].append(naiveUCB1__['regret'][0])
        res_['UCB1']['cumrew_' + str(dif)] = naiveUCB1__['cumreward']
        res_['UCB1']['cumrew'].append(naiveUCB1__['cumreward'][0])
        res_['UCB1']['Reward'].append(naiveUCB1__['reward'][0])

        res_['ADAETC']['regret_' + str(dif)] = ADAETC__['regret']
        res_['ADAETC']['Regret'].append(ADAETC__['regret'][0])
        res_['ADAETC']['cumrew_' + str(dif)] = ADAETC__['cumreward']
        res_['ADAETC']['cumrew'].append(ADAETC__['cumreward'][0])
        res_['ADAETC']['Reward'].append(ADAETC__['reward'][0])

        res_['ETC']['regret_' + str(dif)] = ETC__['regret']
        res_['ETC']['Regret'].append(ETC__['regret'][0])
        res_['ETC']['cumrew_' + str(dif)] = ETC__['cumreward']
        res_['ETC']['cumrew'].append(ETC__['cumreward'][0])
        res_['ETC']['Reward'].append(ETC__['reward'][0])

        res_['NADAETC']['regret_' + str(dif)] = NADAETC__['regret']
        res_['NADAETC']['Regret'].append(NADAETC__['regret'][0])
        res_['NADAETC']['cumrew_' + str(dif)] = NADAETC__['cumreward']
        res_['NADAETC']['cumrew'].append(NADAETC__['cumreward'][0])
        res_['NADAETC']['Reward'].append(NADAETC__['reward'][0])

        res_['UCB1-s']['regret_' + str(dif)] = UCB1_stopping__['regret']
        res_['UCB1-s']['Regret'].append(UCB1_stopping__['regret'][0])
        res_['UCB1-s']['cumrew_' + str(dif)] = UCB1_stopping__['cumreward']
        res_['UCB1-s']['cumrew'].append(UCB1_stopping__['cumreward'][0])
        res_['UCB1-s']['Reward'].append(UCB1_stopping__['reward'][0])

        res_['SuccElim']['regret_' + str(dif)] = SuccElim__['regret']
        res_['SuccElim']['Regret'].append(SuccElim__['regret'][0])
        res_['SuccElim']['cumrew_' + str(dif)] = SuccElim__['cumreward']
        res_['SuccElim']['cumrew'].append(SuccElim__['cumreward'][0])
        res_['SuccElim']['Reward'].append(SuccElim__['reward'][0])
        return res_

    def plot_varying_delta(res_, delt_, numSim, T_, title='Regret'):
        bw = 0.15  # bar width
        naive_ucb1 = res_['UCB1'][title]
        adaetc = res_['ADAETC'][title]
        etc = res_['ETC'][title]
        nadaetc = res_['NADAETC'][title]
        ucb1s = res_['UCB1-s'][title]
        succ_elim = res_['SuccElim'][title]

        bar1 = np.arange(len(naive_ucb1))
        bar2 = [x + bw for x in bar1]
        bar3 = [x + bw for x in bar2]
        bar4 = [x + bw for x in bar3]
        bar5 = [x + bw for x in bar4]
        bar6 = [x + bw for x in bar5]
        plt.figure(figsize=(12, 8), dpi=150)

        plt.bar(bar1, adaetc, color='r', width=bw, edgecolor='grey', label='ADA-ETC')
        plt.bar(bar2, etc, color='g', width=bw, edgecolor='grey', label='ETC')
        plt.bar(bar3, nadaetc, color='maroon', width=bw, edgecolor='grey', label='NADA-ETC')
        plt.bar(bar4, ucb1s, color='navy', width=bw, edgecolor='grey', label='UCB1-s')
        plt.bar(bar5, succ_elim, color='purple', width=bw, edgecolor='grey', label='SuccElim')
        plt.bar(bar6, naive_ucb1, color='b', width=bw, edgecolor='grey', label='UCB1')

        if title == 'cumrew':
            chartTitle = 'Cumulative Reward'
            plt.ylim(ymax=100)
        elif title == 'Reward':
            chartTitle = 'Best Arm Reward'
            plt.ylim(ymax=95)
        elif title == 'Regret':
            chartTitle = 'Regret'
            plt.ylim(ymax=30)
        plt.ylabel(chartTitle, fontsize=15)
        plt.xlabel(r'$\Delta$', fontweight='bold', fontsize=15)
        plt.xticks([x + bw for x in bar1], delt_)

        plt.legend(loc="upper right") if title == 'Regret' else plt.legend(loc="upper left")
        plt.savefig('res/2arms_halfAndHalfPlusDelta_'+title+'_' + str(numSim) + 'sims_T' + str(T_) + '.eps',
                    format='eps', bbox_inches='tight')
        # plt.show()
        plt.cla()

    print("Running m = 1")
    start_ = time.time()
    res = init_res()
    delt = np.zeros(rng)
    for i in range(rng):
        delt[i] = round((1 / ((rng - 1) * 2)) * i, 3)
        armInstances_ = gA.generateTwoArms(T_list_, 1, delta=np.ones(len(T_list_)) * delt[i])
        print("Running m=1")
        naiveUCB1_ = fA.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_, start_)
        ADAETC_ = fA.ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
        NADAETC_ = fA.NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_)
        SuccElim_ = fA.SuccElim(armInstances_, startSim_, endSim_, K_list_, T_list_, constant_c=4)
        res = store_res(res, i, naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_)
    print("took " + str(time.time() - start_) + " seconds")

    plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0])
    plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], 'cumrew')
    plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], 'Reward')

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'ETC': ETC_,
            'NADAETC': NADAETC_,
            'UCB1-s': UCB1_stopping_,
            'SuccElim': SuccElim_,
            'fullRes': res}


def mEqOne(K_list_, T_list_, numArmDists_, startSim_, endSim_, alpha__, pw_):

    print("Running m = 1")
    constant_c = 2
    start_ = time.time()
    armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__)
    # armInstances_ = gA.generateMultipleArms(K_list, T_list, numArmDists, pw_)

    naiveUCB1_ = fA.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_, start_)
    ADAETC_ = fA.ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_)
    ETC_ = fA.ETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
    NADAETC_ = fA.NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
    UCB1_stopping_ = fA.UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_)
    SuccElim_ = fA.SuccElim(armInstances_, startSim_, endSim_, K_list_, T_list_, constant_c)

    print("took " + str(time.time() - start_) + " seconds")

    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.plot(T_list, naiveUCB1_['regret'], color='b', label='UCB1')
    plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                 color='b', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
    plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                 color='r', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, ETC_['regret'], color='g', label='ETC')
    plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                 color='g', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, NADAETC_['regret'], color='maroon', label='NADA-ETC')
    plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                 color='maroon', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='UCB1-s')
    plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                 color='navy', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, SuccElim_['regret'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
    plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                 color='purple', fmt='o', markersize=4, capsize=4)
    plt.xticks(T_list)
    # plt.ylim(ymax=30)
    plt.ylabel('Regret', fontsize=15)
    plt.xlabel('T', fontsize=15)

    plt.legend(loc="upper left")
    plt.savefig('res/mEquals1_K' + str(K_list_[0]) + '_alpha' + str(alpha__) + '_sim' + str(endSim_ - startSim_) +
                '_armDist' + str(numArmDists_) + '_c' + str(constant_c) + '.eps', format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'ETC': ETC_,
            'NADAETC': NADAETC_,
            'UCB1-s': UCB1_stopping_,
            'SuccElim': SuccElim_}


def mGeneral(K_list_, T_list_, numArmDists_, startSim_, endSim_, m_, alpha__, pw_):
    print("Running m =", m_)
    start_ = time.time()
    armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__)
    # armInstances_ = gA.generateMultipleArms(K_list_, T_list_, numArmDists_, pw_)
    print("Running m =", m_)
    RADAETC_ = fA.RADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_NADAETC_ = fA.m_NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    print("took " + str(time.time() - start_) + " seconds")

    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.plot(T_list, m_ADAETC_['regret'], color='r', label='m-ADA-ETC')
    plt.errorbar(T_list, m_ADAETC_['regret'], yerr=m_ADAETC_['standardError'],
                 color='r', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, m_ETC_['regret'], color='g', label='m-ETC')
    plt.errorbar(T_list, m_ETC_['regret'], yerr=m_ETC_['standardError'],
                 color='g', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, m_NADAETC_['regret'], color='maroon', label='m-NADA-ETC')
    plt.errorbar(T_list, m_NADAETC_['regret'], yerr=m_NADAETC_['standardError'],
                 color='maroon', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, m_UCB1_stopping_['regret'], color='navy', label='m-UCB1-s')
    plt.errorbar(T_list, m_UCB1_stopping_['regret'], yerr=m_UCB1_stopping_['standardError'],
                 color='navy', fmt='o', markersize=4, capsize=4)
    plt.plot(T_list, RADAETC_['regret'], color='purple', label='RADA-ETC')
    plt.errorbar(T_list, RADAETC_['regret'], yerr=RADAETC_['standardError'],
                 color='purple', fmt='o', markersize=4, capsize=4)
    plt.xticks(T_list)
    # plt.ylim(ymax=30)
    plt.ylabel('Regret', fontsize=15)
    plt.xlabel('T', fontsize=15)

    plt.legend(loc="upper left")
    plt.savefig('res/mEquals' + str(m_) + '_K' + str(K_list_[0]) + '_alpha' + str(alpha__) + '_sim' +
                str(endSim_ - startSim_) + '_armDist' + str(numArmDists_) + '.eps', format='eps', bbox_inches='tight')
    plt.show()
    plt.cla()

    return {'RADAETC': RADAETC_,
            'ADAETC': m_ADAETC_,
            'ETC': m_ETC_,
            'NADAETC': m_NADAETC_,
            'UCB1-s': m_UCB1_stopping_}


if __name__ == '__main__':
    K_list = np.array([2])
    # varyingK = True if len(K_list) > 1 else False
    T_list = np.array([100])  # np.arange(1, 6) * 100  #
    m = 5
    numArmDists = 1  # 100
    alpha_ = 0  # can be used for both
    # beta_ = 0.01  # for rotting bandits
    startSim = 0
    endSim = 10000
    pw = 1 / 2  # used for both, larger pw means higher variance in mean changes for rotting
    # larger pw means closer mean rewards in the arm instances generated

    # market-like simulation
    # meanK = 10  # we will have totalPeriods-many streams, so mean can be set based on that
    # meanT = 200
    # numStreams = 1  # number of different K & T streams in total
    # totalPeriods = 40
    # m_vals = np.array([1, 2, 4, 5, 8, 10, 20, 40])
    # market = marketSim(meanK, meanT, numArmDists, numStreams, totalPeriods, m_vals,
    #                    alpha_, startSim, endSim, oneOptPerPeriod=True)
    # exit()

    # rotting bandits part
    # rotting(K_list, T_list, numArmDists, alpha_, beta_, startSim, endSim, pw)
    # exit()

    # fixed mean rewards throughout, m = 1
    # result = mEqOne(K_list, T_list, numArmDists, startSim, endSim, alpha_, pw)
    # exit()

    # fixed means but two arms, varying difference between means, m = 1
    mEqOne_2arms(K_list, T_list, startSim, endSim, rng=11)
    exit()

    # fixed mean rewards throughout, m > 1
    mGeneral(K_list, T_list, numArmDists, startSim, endSim, m, alpha_, pw)
    exit()

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
