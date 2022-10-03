import time
import generateArms as gA
import fixedArmsRotting as fAR
from supportingMethods import *
from copy import deepcopy


def marketSim(meanK_, meanT_, numArmDists_, numStreams_, totalPeriods_, m_vals_,
              alpha__, endSim_, algs=None, ucbPart_=2, numOptPerPeriod='none'):
    if algs is None:
        algs = {'ada': {}, 'rada': {}, 'nada': {}, 'ucb1s': {}, 'ucb1': {}, 'etc': {}}
    start_ = time.time()
    res = {}  # used to store the reward for each (K, T)-pair stream and algorithm
    resreg = {}  # used to store the regret for each (K, T)-pair stream and algorithm
    stdev = {}  # used to store the standard deviation of rewards for each (K, T)-pair stream and algorithm
    best_rew = {}  # used to store the best reward for each m value and instance

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
        res['stream_' + str(st)] = {}
        resreg['stream_' + str(st)] = {}
        stdev['stream_' + str(st)] = {}
        best_rew['stream_' + str(st)] = {}

        K_list_ = np.array(K_list_stream[str(st)])
        T_list_ = np.array(T_list_stream[str(st)])

        for a in range(numArmDists_):
            res['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            resreg['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            stdev['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            best_rew['stream_' + str(st)]['arm_' + str(a)] = np.zeros(len(m_vals_))

            if (a + 1) % 5 == 0:
                print("Arm dist number ", str(a + 1), ", K&T stream number", str(st + 1), ", time ",
                      str(time.time() - start_), " from start.")

            # generate arm means, either each period has one optimal arm (across all periods) or no such arrangement
            armInstances_ = gA.generateArms_marketSim(K_list_, T_list_, totalPeriods_, alpha__, numOptPerPeriod)['arms']

            # run multiple simulations for varying values of m
            # if m = 1, every period is run individually and ADA-ETC is called using corresponding (K, T) pair
            # if m > 1, then multiple periods are combined and m-ADA-ETC is called using corresponding (K, T) pair which
            # for this case is summation of different K's and T's.
            for i in range(len(m_vals_)):
                # decide on the number of calls to simulation module and the corresponding m value
                m_val = m_vals_[i]  # say this is 3

                stdev_local = {}
                for keys in algs.keys():
                    res['stream_' + str(st)]['arm_' + str(a)][keys]['m = ' + str(m_val)] = 0
                    resreg['stream_' + str(st)]['arm_' + str(a)][keys]['reg: m = ' + str(m_val)] = 0
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

                    # get the best reward for this m value, and (K, T)-pair
                    best_rew['stream_' + str(st)]['arm_' + str(a)][i] += \
                        np.mean(np.sort(arms_[0])[-m_val:]) * T_vals_ / totalPeriods_

                    # run the multiple simulations on different algorithms
                    for alg_keys in algs.keys():
                        for t in range(endSim_):
                            run = sim_small_mid_large_m(arms_, np.array([K_vals_]), np.array([T_vals_]),
                                                        m_val, ucbPart_, alg=alg_keys)
                            rew = run['reward'].item()
                            stdev_local[alg_keys].append(rew)
                            rew /= (calls * int(endSim_))
                            reg = run['regret'].item() / (calls * int(endSim_))
                            # .item() is used to get the value, not as a 1-dim array, i.e., fixing the type
                            res['stream_' + str(st)]['arm_' + str(a)][alg_keys]['m = ' + str(m_val)] += rew
                            resreg['stream_' + str(st)]['arm_' + str(a)][alg_keys]['reg: m = ' + str(m_val)] += reg

                    # get the standard deviation on rewards
                    for alg_keys in algs.keys():
                        stdev['stream_' + str(st)]['arm_' + str(a)][alg_keys]['m = ' + str(m_val)] = \
                            np.sqrt(np.var(np.array(stdev_local[alg_keys])) / int(len(stdev_local[alg_keys])))

    # storing results for each algorithm for each m value, across different (K, T)-pair streams (if multiple)
    rewards, stdevs, regrets = deepcopy(algs), deepcopy(algs), deepcopy(algs)
    best_rews_by_m = np.zeros(len(m_vals_))
    normalizer = numArmDists_ * numStreams_
    for j in range(len(m_vals_)):
        m_val = m_vals_[j]
        for keys in algs.keys():
            rewards[keys]['m = ' + str(m_val)] = 0
            regrets[keys]['reg: m = ' + str(m_val)] = 0
            stdevs[keys]['m = ' + str(m_val)] = 0
        for st in range(numStreams_):
            for a in range(numArmDists_):
                best_rews_by_m[j] += best_rew['stream_' + str(st)]['arm_' + str(a)][j] / normalizer
                for kys in algs.keys():
                    rewards[kys]['m = ' + str(m_val)] += \
                        res['stream_' + str(st)]['arm_' + str(a)][kys]['m = ' + str(m_val)] / normalizer
                    regrets[kys]['reg: m = ' + str(m_val)] += \
                        resreg['stream_' + str(st)]['arm_' + str(a)][kys]['reg: m = ' + str(m_val)] / normalizer
                    stdevs[kys]['m = ' + str(m_val)] += \
                        stdev['stream_' + str(st)]['arm_' + str(a)][kys]['m = ' + str(m_val)] / np.sqrt(normalizer)

    # used to pass the rewards and standard deviations to the plotting module as arrays
    alg_rews = {'ada': [], 'rada': [], 'etc': [], 'nada': [], 'ucb1s': [], 'ucb1': []}
    alg_stdevs = {'ada': [], 'rada': [], 'etc': [], 'nada': [], 'ucb1s': [], 'ucb1': []}

    # printing results for each algorithm
    print("numArmDist", numArmDists_, "; alpha", alpha__, "; sims", endSim_,
          "; meanK", meanK_, "; meanT", meanT_, "; numStreams", numStreams_)
    print("How many optimal arm(s) per period?", numOptPerPeriod, "; total periods", totalPeriods_)
    for keys in algs.keys():
        print('--' * 20)
        print(keys + ' results ')
        print('Rewards:')
        print('Best: ', best_rews_by_m)
        for i in range(len(m_vals_)):
            rew = round(rewards[keys]['m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", rew)
            alg_rews[keys].append(rew)
        print()
        print('Standard deviation of rewards')
        for i in range(len(m_vals_)):
            stdev = round(stdevs[keys]['m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", stdev)
            alg_stdevs[keys].append(stdev)
        print()
        print('Regrets')
        for i in range(len(m_vals_)):
            reg = round(regrets[keys]['reg: m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", reg)

    print("Done after ", str(round(time.time() - start_, 2)), " seconds from start.")
    params_ = {'numOpt': numOptPerPeriod, 'alpha': alpha__, 'bestReward': best_rews_by_m,
               'numArmDists': numArmDists_, 'totalSim': endSim_}

    plot_marketSim(meanK_, meanT_, m_vals_, alg_rews, alg_stdevs, params_)


def rotting(K_list_, T_list_, numArmDists_, endSim_, alpha__, beta__, pw_):
    print("Running rotting bandits")
    start_ = time.time()
    armInstances_ = gA.generateRottingArms(K_list_[0], T_list_, numArmDists_, alpha__, beta__)
    print("Running rotting bandits")

    fAR.naiveUCB1(armInstances_, endSim_, K_list_, T_list_, pw_)
    fAR.ADAETC(armInstances_, endSim_, K_list_, T_list_, pw_)
    fAR.Rotting(armInstances_, endSim_, K_list_, T_list_, pw_,
                           sigma=1, deltaZero=2, alpha=0.1)
    print("took " + str(time.time() - start_) + " seconds; alpha " + str(alpha__) + ", beta " + str(beta__))


def mEqOne_barPlots(K_list_, T_list_, endSim_, alpha__, numOpt_, generateIns_,
                    rng=11, ucbSim=True, justUCB='no'):
    print("Running m = 1, justUCB: " + justUCB)
    start_ = time.time()
    res = init_res()
    delt = np.zeros(rng)

    for i in range(rng):
        multi = 50 if justUCB == 'yes' else 1
        print('Iteration ', i)
        delt[i] = round((1 / ((rng - 1) * 2 * multi)) * i, 5)
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, generateIns_, alpha__, numOpt_,
                                                   delta=np.array([delt[i]]), verbose=True)
        if justUCB == 'yes':
            T_list_ = np.array([int(1 / delt[i])]) if delt[i] > 0 else np.array([100])
            # used to test for 0.5 v. 0.5 + 1/T case

        if ucbSim or (justUCB == 'yes'):
            naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_)
            res = store_res(res, generateIns_, i, naiveUCB1_, 'UCB1')
        if justUCB == 'no':
            ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
            res = store_res(res, generateIns_, i, ADAETC_, 'ADAETC')
            ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
            res = store_res(res, generateIns_, i, ETC_, 'ETC')
            NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
            res = store_res(res, generateIns_, i, NADAETC_, 'NADAETC')
            UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_)
            res = store_res(res, generateIns_, i, UCB1_stopping_, 'UCB1-s')
            SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c=4)
            res = store_res(res, generateIns_, i, SuccElim_, 'SuccElim')
    print("took " + str(time.time() - start_) + " seconds")

    if justUCB == 'no':
        a = 0 if ucbSim else 1
        for i in range(a, 2):
            UCBin = i == 0
            plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                               generateIns_, alpha__, numOpt_, UCBin)
            plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                               generateIns_, alpha__, numOpt_, UCBin, 'cumrew')
            plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                               generateIns_, alpha__, numOpt_, UCBin, 'cumReg')
            plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                               generateIns_, alpha__, numOpt_, UCBin, 'Reward')


def mEqOne(K_list_, T_list_, numArmDists_, endSim_, alpha__, numOpt_, delt_,
           plots=True, ucbSim=True, fixed='whatever', justUCB='no', Switch='no'):
    print("Running m = 1, justUCB: " + justUCB + ", Switch: " + Switch)
    constant_c = 4
    start_ = time.time()
    if fixed == 'Gap':
        armInstances_ = gA.generateArms_fixedGap(K_list, T_list, numArmDists_, verbose=True)  # deterministic
        print("Fixed gaps")
    elif fixed == 'Intervals':
        armInstances_ = gA.generateArms_fixedIntervals(K_list, T_list, numArmDists_, verbose=True)  # randomized
        print("Fixed intervals")
    elif justUCB == 'yes':
        def fn(x):
            return (1 / np.sqrt(x)).round(5)
        delt = fn(T_list_)
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__, numOpt_,
                                                   delta=delt, verbose=True)
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
    naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, Switch_ = None, None, None, None, None, None, None
    if justUCB == 'no':
        ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
        NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_)
        SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c)
    if ucbSim:
        naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_)
    if Switch == 'yes':
        Switch_ = fA.Switching(armInstances_, endSim, K_list, T_list)

    print("took " + str(time.time() - start_) + " seconds")
    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_, 'm': 1}
    _ = None
    if plots:
        a = 0 if ucbSim else 1
        if justUCB == 'no':
            for i in range(a, 2):
                plot_fixed_m(i, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_,
                             NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if ucbSim and (justUCB == 'no'):
            plot_fixed_m(3, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_,
                         NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if Switch == 'yes':
            plot_fixed_m(4, K_list_, T_list, naiveUCB1_, _, _, _, _, _, Switch_, params_)
            plot_fixed_m(5, K_list_, T_list, naiveUCB1_, _, _, _, _, _, Switch_, params_)


def mGeneral(K_list_, T_list_, numArmDists_, endSim_, m_, alpha__, numOpt_, delt_):
    print("Running m =", m_)
    start_ = time.time()
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=False)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__,
                                                   numOpt_, delt_, verbose=False)
        print(str(numOpt_) + ' optimal arms')
    print("Running m =", m_)
    RADAETC_ = fA.RADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, endSim, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_NADAETC_ = fA.m_NADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, m_)
    m_naiveUCB1 = fA.m_naiveUCB1(armInstances_, endSim_, K_list_, T_list_, m_)
    print("took " + str(time.time() - start_) + " seconds")

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': 4, 'delta': delt_, 'm': m_}

    # first argument is set to 2 to use the general m plots
    plot_fixed_m(2, K_list_, T_list, m_naiveUCB1, m_ADAETC_, m_ETC_, m_NADAETC_, m_UCB1_stopping_, RADAETC_,
                 RADAETC_, params_)  # last RADAETC_ is for switching bandits


if __name__ == '__main__':
    K_list = np.array([2])
    T_list = np.arange(1, 16) * 1000  # np.arange(1, 3) * 250000  # np.array([100])  #
    m = 2
    numArmDists = 25
    alpha_ = 0.4
    ucbPart = 2
    endSim = 20
    doing = 'market'  # 'm1', 'mGeq1', 'm1bar', 'market', 'rott'

    if doing == 'market':
        # market-like simulation
        meanK = 10  # we will have totalPeriods-many streams, so mean can be set based on that
        meanT = 100
        numStreams = 1  # number of different K & T streams in total
        algorithms = None  # go for {'rada': {}, 'ucb1': {}} if only want rada-etc and ucb1
        totalPeriods = 120
        m_vals = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
        # np.array([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72])
        marketSim(meanK, meanT, numArmDists, numStreams, totalPeriods, m_vals, alpha_, endSim,
                  algs=algorithms, ucbPart_=ucbPart, numOptPerPeriod='none')  # or any other integer
    elif doing == 'm1':
        # fixed mean rewards throughout, m = 1
        mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
               numOpt_=1, delt_=0, plots=True, ucbSim=True, fixed='no', justUCB='yes', Switch='yes')
        # # fixed='Intervals' or 'Gap' or anything else
    elif doing == 'mGeq1':
        # fixed mean rewards throughout, m > 1
        mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_,
                 numOpt_=3, delt_=0.3)
    elif doing == 'm1bar':
        # fixed means but difference is specified between two best arms, varying difference between means, m = 1
        mEqOne_barPlots(K_list, T_list, endSim, alpha_,
                        numOpt_=1, generateIns_=numArmDists, rng=11, ucbSim=True, justUCB='yes')
        # generateIns_ takes the place of running with multiple arm instances
        # It is needed if K > 2, because then we will be generating K - 2 random arms in uniform(0, 0.5)
        # change numOpt to >1 to generate K - numOpt - 1 random arms in uniform(0, 0.5), one at exactly 0.5,
        # and numOpt many with delta distance to 0.5
        # ucbSim is True is going for including UCB in
        # justUCB is 'no' if not going only for UCB but others too, o/w it's 'yes'
        # If yes, the instances are so that the \Delta between arms govern the T
    elif doing == 'rott':
        # rotting bandits part
        rotting(K_list, T_list, numArmDists, endSim, alpha_,
                beta__=0.01, pw_=1 / 2)
        # larger pw means higher variance in mean changes
