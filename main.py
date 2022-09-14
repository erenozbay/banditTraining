# import numpy as np
# import pandas as pd
# from scipy.stats import nbinom
# import matplotlib.pyplot as plt
# import fixedArms as fA
import time
import generateArms as gA
import fixedArmsRotting as fAR
from supportingMethods import *
from copy import deepcopy


# totalPeriods_ should be divisible by all values in m_vals, e.g., 12 periods, m_vals = [1, 2, 3, 6]
def marketSim(meanK_, meanT_, numArmDists_, numStreams_, totalPeriods_, m_vals_,
              alpha__, startSim_, endSim_, ucbPart_=2, numOptPerPeriod='none'):
    start_ = time.time()
    res = {}  # used to store the reward for each (K, T)-pair stream and algorithm
    resreg = {}  # used to store the regret for each (K, T)-pair stream and algorithm
    stdev = {}  # used to store the standard deviation of rewards for each (K, T)-pair stream and algorithm
    algs = {'ada': {}, 'rada': {}, 'nada': {}, 'ucb1s': {}, 'etc': {}}
    algs_for_res = deepcopy(algs)
    algs_for_res.update(dict(best=0))
    for st in range(numStreams_):
        res['stream_' + str(st)] = algs_for_res
        resreg['stream_' + str(st)] = deepcopy(algs)
        stdev['stream_' + str(st)] = deepcopy(algs)

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
    sampleStreams = sample_K_T_streams(numStreams_, totalPeriods_, meanK_, meanT_, m_vals_, geom=False)
    K_list_stream = sampleStreams['K_list_stream']
    T_list_stream = sampleStreams['T_list_stream']
    print("GO")

    # run the simulation for each (K, T)-pair
    # fix (K, T) and following it, fix an instance with the correct number of arms, following the period-based K values
    # here we have the option of (1) either generation one optimal arm per period (by keeping oneOptPerPeriod = True)
    # which generates all the required arms across all periods, selects the best totalPeriods_-many of them and places
    # one of each to every period, and the rest of the arms required for the periods are placed randomly; or
    # (2) generate everything randomly
    for st in range(numStreams_):
        K_list_ = np.array(K_list_stream[str(st)])
        T_list_ = np.array(T_list_stream[str(st)])
        # print(K_list_)
        for a in range(numArmDists_):
            if (a + 1) % 5 == 0:
                print("Arm dist number ", str(a + 1), ", K&T stream number", str(st + 1), ", time ",
                      str(time.time() - start_), " from start.")

            # generate arm means, either each period has one optimal arm (across all periods) or no such arrangement
            armsInfo = gA.generateArms_marketSim(K_list_, T_list_, totalPeriods_, alpha__, numOptPerPeriod)
            armInstances_ = armsInfo['armInstances']
            best_top_m_avg_reward = armsInfo['best_top_m_avg_reward']
            total_T = armsInfo['total_T']
            # print(armInstances_)
            # run multiple simulations for varying values of m
            # if m = 1, every period is run individually and ADA-ETC is called using corresponding (K, T) pair
            # if m > 1, then multiple periods are combined and m-ADA-ETC is called using corresponding (K, T) pair which
            # for this case is summation of different K's and T's.
            for i in range(len(m_vals_)):
                # decide on the number of calls to simulation module and the corresponding m value
                m_val = m_vals_[i]  # say this is 3

                stdev_local = {}
                for keys in algs.keys():
                    res['stream_' + str(st)][keys]['m = ' + str(m_val)] = 0
                    resreg['stream_' + str(st)][keys]['reg: m = ' + str(m_val)] = 0
                    stdev_local[keys] = []

                calls = int(totalPeriods_ / m_val)  # and this is 6 / 3 = 2, i.e., totalPeriods_ = 6
                for p in range(calls):  # so this will be 0, 1
                    indices = np.zeros(m_val)  # should look like 0, 1, 2;        3,  4,  5
                    for j in range(m_val):
                        indices[j] = p * m_val + j  # so this does the below example
                        # mval = 3; p = 0, j = 0 => 0; p = 0, j = 1 => 1; p = 0, j = 2 => 2
                        # mval = 3; p = 1, j = 0 => 3; p = 1, j = 1 => 4; p = 1, j = 2 => 5
                    # properly combine K and T values if needed
                    K_vals_, T_vals_ = 0, 0
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
                    # print(arms_)
                    # run the multiple simulations on different algorithms
                    for alg_keys in algs.keys():
                        for t in range(endSim_ - startSim_):
                            run = sim_small_mid_large_m(arms_, np.array([K_vals_]), np.array([T_vals_]),
                                                        m_val, ucbPart_, alg=alg_keys)
                            rew = run['reward'].item()
                            stdev_local[alg_keys].append(rew)
                            rew /= (calls * int(endSim_ - startSim_))
                            reg = run['regret'].item() / (calls * int(endSim_ - startSim_))
                            # .item() is used to get the value, not as array
                            res['stream_' + str(st)][alg_keys]['m = ' + str(m_val)] += rew
                            resreg['stream_' + str(st)][alg_keys]['reg: m = ' + str(m_val)] += reg

                    # get the standard deviation on rewards
                    for alg_keys in algs.keys():
                        stdev['stream_' + str(st)][alg_keys]['m = ' + str(m_val)] = \
                            np.sqrt(np.var(np.array(stdev_local[alg_keys])) / int(len(stdev_local[alg_keys])))

            # store the best reward
            res['stream_' + str(st)]['best'] += total_T * best_top_m_avg_reward / numArmDists_

    # storing results for each algorithm for each m value, across different (K, T)-pair streams (if multiple)
    rewards, stdevs, regrets, best_rewards, once = deepcopy(algs), deepcopy(algs), deepcopy(algs), 0, 0
    for j in range(len(m_vals_)):
        m_val = m_vals_[j]
        for keys in algs.keys():
            rewards[keys]['m = ' + str(m_val)] = 0
            regrets[keys]['reg: m = ' + str(m_val)] = 0
            stdevs[keys]['m = ' + str(m_val)] = 0
        for st in range(numStreams_):
            for kys in algs.keys():
                rewards[kys]['m = ' + str(m_val)] += res['stream_' + str(st)][kys]['m = ' + str(m_val)] / numStreams_
                regrets[kys]['reg: m = ' + str(m_val)] += \
                    resreg['stream_' + str(st)][kys]['reg: m = ' + str(m_val)] / numStreams_
                stdevs[kys]['m = ' + str(m_val)] += stdev['stream_' + str(st)][kys]['m = ' + str(m_val)] / numStreams_
            if once < numStreams_:
                best_rewards += res['stream_' + str(st)]['best']
                once += 1

    # printing results for each algorithm
    print("numArmDist", numArmDists_, "; alpha", alpha__, "; sims", endSim_ - startSim_,
          "; meanK", meanK_, "; meanT", meanT_, "; numStreams", numStreams_)
    print("How many optimal arm(s) per period?", numOptPerPeriod, "; total periods", totalPeriods_)
    for keys in algs.keys():
        print('--' * 20)
        print(keys + ' results ')
        print('Rewards:')
        print('Best: ', best_rewards)
        for i in range(len(m_vals_)):
            print("m =", str(m_vals_[i]), ": ", round(rewards[keys]['m = ' + str(m_vals_[i])], 5))
        print()
        print('Standard deviation of rewards')
        for i in range(len(m_vals_)):
            print("m =", str(m_vals_[i]), ": ", round(stdevs[keys]['m = ' + str(m_vals_[i])], 5))
        print()
        print('Regrets')
        for i in range(len(m_vals_)):
            print("m =", str(m_vals_[i]), ": ", round(regrets[keys]['reg: m = ' + str(m_vals_[i])], 5))
    print("UCB part for NADA-ETC " + str(ucbPart_) + ", and UCB1-s is " + str(ucbPart_))
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


def mEqOne_barPlots(K_list_, T_list_, startSim_, endSim_, alpha__, ucbPart_, numOpt_, generateIns_, rng=11):
    print("Running m = 1")
    start_ = time.time()
    res = init_res()
    delt = np.zeros(rng)
    naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, SuccElim6_ = 0, 0, 0, 0, 0, 0, 0
    for i in range(rng):
        print('Iteration ', i)
        delt[i] = round((1 / ((rng - 1) * 2)) * i, 3)
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, generateIns_, alpha__, numOpt_,
                                                   delta=delt[i], verbose=False)
        naiveUCB1_ = fA.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_)
        res = store_res(res, generateIns_, i, naiveUCB1_, 'UCB1')
        ADAETC_ = fA.ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_)
        res = store_res(res, generateIns_, i, ADAETC_, 'ADAETC')
        ETC_ = fA.ETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
        res = store_res(res, generateIns_, i, ETC_, 'ETC')
        NADAETC_ = fA.NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, ucbPart_)
        res = store_res(res, generateIns_, i, NADAETC_, 'NADAETC')
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_, ucbPart_)
        res = store_res(res, generateIns_, i, UCB1_stopping_, 'UCB1-s')
        SuccElim_ = fA.SuccElim(armInstances_, startSim_, endSim_, K_list_, T_list_, constant_c=4)
        res = store_res(res, generateIns_, i, SuccElim_, 'SuccElim')
    print("took " + str(time.time() - start_) + " seconds")

    for i in range(2):
        UCBin = i == 0
        plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], K_list[0],
                           generateIns_, alpha__, numOpt_, UCBin)
        plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], K_list[0],
                           generateIns_, alpha__, numOpt_, UCBin, 'cumrew')
        plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], K_list[0],
                           generateIns_, alpha__, numOpt_, UCBin, 'cumReg')
        plot_varying_delta(res, delt, endSim_ - startSim_, T_list_[0], K_list[0],
                           generateIns_, alpha__, numOpt_, UCBin, 'Reward')

    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'ETC': ETC_,
            'NADAETC': NADAETC_,
            'UCB1-s': UCB1_stopping_,
            'SuccElim': SuccElim_,
            'fullRes': res}


def mEqOne(K_list_, T_list_, numArmDists_, startSim_, endSim_, alpha__, ucbPart_, numOpt_, delt_,
           plots=True, fixed='whatever'):
    print("Running m = 1")
    constant_c = 4
    start_ = time.time()
    if fixed == 'Gap':
        armInstances_ = gA.generateArms_fixedGap(K_list, T_list, numArmDists_, verbose=True)  # deterministic
        print("Fixed gaps")
    elif fixed == 'Intervals':
        armInstances_ = gA.generateArms_fixedIntervals(K_list, T_list, numArmDists_, verbose=True)  # randomized
        print("Fixed intervals")
    else:
        if numOpt_ == 1 and delt_ == 0:
            armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
            print("Single optimal arm with random gap")
        else:
            armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__,
                                                       numOpt_, delt_, verbose=True)
            if numOpt_ == 1:
                print("Single optimal arm with gap " + str(delt_))
            else:
                print(str(numOpt_) + " opt arms, gap " + str(delt_))

    ADAETC_ = fA.ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_)
    ETC_ = fA.ETC(armInstances_, startSim_, endSim_, K_list_, T_list_)
    NADAETC_ = fA.NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, ucbPart_)
    UCB1_stopping_ = fA.UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_, ucbPart_, orig=True)
    SuccElim_ = fA.SuccElim(armInstances_, startSim_, endSim_, K_list_, T_list_, constant_c)
    naiveUCB1_ = fA.naiveUCB1(armInstances_, startSim_, endSim_, K_list_, T_list_)

    print("took " + str(time.time() - start_) + " seconds")

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_ - startSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_, 'm': 1, 'ucbPart': ucbPart_}
    if plots:
        for i in range(2):
            plot_fixed_m(i, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, params_)
        plot_fixed_m(3, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, params_)
    print("UCB part for NADA-ETC " + str(ucbPart_) + ", and UCB1-s is " + str(ucbPart_))
    return {'UCB1': naiveUCB1_,
            'ADAETC': ADAETC_,
            'ETC': ETC_,
            'NADAETC': NADAETC_,
            'UCB1-s': UCB1_stopping_,
            'SuccElim': SuccElim_}


def mGeneral(K_list_, T_list_, numArmDists_, startSim_, endSim_, m_, alpha__, ucbPart_, numOpt_, delt_):
    print("Running m =", m_)
    start_ = time.time()
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=False)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__,
                                                   numOpt_, delt_, verbose=False)
        print(str(numOpt_) + ' optimal arms')
    print("Running m =", m_)
    RADAETC_ = fA.RADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, startSim_, endSim, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_)
    m_NADAETC_ = fA.m_NADAETC(armInstances_, startSim_, endSim_, K_list_, T_list_, m_, ucbPart_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, startSim_, endSim_, K_list_, T_list_, m_, ucbPart_, orig=True)
    print("took " + str(time.time() - start_) + " seconds")

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_ - startSim_,
               'numArmDists': numArmDists_, 'c': 1, 'delta': delt_, 'm': m_, 'ucbPart': ucbPart_}
    # first argument is set to 2 to use general m plots, fourth argument has no effect here
    plot_fixed_m(2, K_list_, T_list, m_ADAETC_, m_ADAETC_, m_ETC_, m_NADAETC_, m_UCB1_stopping_, RADAETC_, params_)
    print("UCB part for NADA-ETC and UCB1-s is " + str(ucbPart_))
    return {'RADAETC': RADAETC_,
            'ADAETC': m_ADAETC_,
            'ETC': m_ETC_,
            'NADAETC': m_NADAETC_,
            'UCB1-s': m_UCB1_stopping_}


if __name__ == '__main__':
    K_list = np.array([2])
    # varyingK = True if len(K_list) > 1 else False
    T_list = np.array([5000])  # np.arange(1, 11) * 1000  # np.array([100])  #
    m = 2
    numArmDists = 1
    alpha_ = 0  # can be used for both
    ucbPart = 2
    # beta_ = 0.01  # for rotting bandits
    startSim = 0
    endSim = 2000
    pw = 1 / 2  # used for both, larger pw means higher variance in mean changes for rotting
    # larger pw means closer mean rewards in the arm instances generated

    # market-like simulation
    # meanK = 3  # we will have totalPeriods-many streams, so mean can be set based on that
    # meanT = 200
    # numStreams = 1  # number of different K & T streams in total
    # totalPeriods = 72
    # m_vals = np.array([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72])
        # np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
    # market = marketSim(meanK, meanT, numArmDists, numStreams, totalPeriods, m_vals,
    #                    alpha_, startSim, endSim, ucbPart_=ucbPart, numOptPerPeriod='none')  # or 'one' or any number
    # exit()

    # fixed mean rewards throughout, m = 1
    # for j in range(5):
    #     result = mEqOne(K_list, T_list, numArmDists, startSim, endSim, alpha_,
    #                     ucbPart_=ucbPart, numOpt_=1, delt_=round(0.1 * (j + 1), 1), plots=True, fixedGap=False)
    # result = mEqOne(K_list, T_list, numArmDists, startSim, endSim, alpha_, ucbPart_=ucbPart,
    #                 numOpt_=1, delt_=0, plots=True, fixed='Intervals')  # fixed='Intervals' or 'Gap' or anything else
    # exit()

    # fixed mean rewards throughout, m > 1
    # mGeneral(K_list, T_list, numArmDists, startSim, endSim, m, alpha_, ucbPart_=ucbPart, numOpt_=3, delt_=0.3)
    # exit()

    # fixed means but difference is specified between two best arms, varying difference between means, m = 1
    mEqOne_barPlots(K_list, T_list, startSim, endSim, alpha_, ucbPart_=ucbPart,
                    numOpt_=1, generateIns_=numArmDists, rng=11)
    exit()
    # generateIns_ takes the place of running with multiple arm instances
    # It is needed if K > 2, because then we will be generating K - 2 random arms in uniform(0, 0.5)
    # change numOpt to >1 to generate K - numOpt - 1 random arms in uniform(0, 0.5), one at exactly 0.5,
    # and numOpt many with delta distance to 0.5

    # rotting bandits part
    # rotting(K_list, T_list, numArmDists, alpha_, beta_, startSim, endSim, pw)
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
