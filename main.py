import time
import generateArms as gA
import fixedArmsRotting as fAR
from supportingMethods import *
from copy import deepcopy
import fixedArms as fA
from datetime import datetime
from manualFigs import *

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
    T_list_raw = deepcopy(T_list_)
    naiveUCB1_, TS, ADAETC_, ETC_, NADAETC_, BAI_ETC, \
        UCB1_stopping_, SuccElim_, Switch_ = None, None, None, None, None, None, None, None, None
    for t in range(len(T_list_raw)):
        T_list_ = np.array([T_list_raw[t]])

        print("Running m = 1, justUCB: " + justUCB)
        start_ = time.time()
        res = init_res()
        delt = np.zeros(rng)

        for i in range(rng):
            multi = 50 if justUCB == 'yes' else 1
            print('Iteration ', i)
            delt[i] = round((1 / ((rng - 1) * 2 * multi)) * i, 5)
            armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, generateIns_, alpha__, numOpt_,
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
                # NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
                res = store_res(res, generateIns_, i, NADAETC_, 'NADAETC')
                UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_)
                res = store_res(res, generateIns_, i, UCB1_stopping_, 'UCB1-s')
                # SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c=4)
                res = store_res(res, generateIns_, i, SuccElim_, 'SuccElim')
        print("took " + str(time.time() - start_) + " seconds")

        if justUCB == 'no':
            a = 0 if ucbSim else 1
            for i in range(a, 2):
                UCBin = i == 0
                plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                                   generateIns_, alpha__, numOpt_, UCBin)
                if len(T_list_raw) == 1:
                    plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                                       generateIns_, alpha__, numOpt_, UCBin, 'cumrew')
                    plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                                       generateIns_, alpha__, numOpt_, UCBin, 'cumReg')
                    plot_varying_delta(res, delt, endSim_, T_list_[0], K_list[0],
                                       generateIns_, alpha__, numOpt_, UCBin, 'Reward')


def mEqOne(K_list_, T_list_, numArmDists_, endSim_, alpha__, numOpt_, delt_,
           plots=True, ucbSim=True, improved=False, fixed='whatever', justUCB='no', Switch='no', NADA='yes'):
    print("Running m = 1, justUCB: " + justUCB + ", Switch: " + Switch + ", NADA: " + NADA, " improved?", improved)
    constant_c = 4
    start_ = time.time()
    if fixed == 'Gap':
        armInstances_ = gA.generateArms_fixedGap(K_list, T_list, numArmDists_, verbose=True)  # deterministic
        print("Fixed gaps")
    elif fixed == 'Intervals':
        armInstances_ = gA.generateArms_fixedIntervals(K_list, T_list, numArmDists_, verbose=True)  # randomized
        print("Fixed intervals")
    elif fixed == 'Delta':
        delt = (np.ones(len(T_list_)) + 3) / 20
        # delt = np.array([delt] * len(T_list_))
        # can make alpha = 0.5 here to get all suboptimal arms at 0.5 and the single optimal arm at 0.5 + delta
        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, alpha__, numOpt_,
                                                   delta=delt, verbose=True)
    elif justUCB == 'yes':
        def fn(x):
            return (1 / np.power(x, 1 / 2)).round(5)

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
    naiveUCB1_, TS, ADAETC_, ETC_, NADAETC_, BAI_ETC, \
        UCB1_stopping_, SuccElim_, Switch_ = None, None, None, None, None, None, None, None, None
    if justUCB == 'no':
        ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
        # if NADA == 'yes':
        #     NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, improved=improved, ucbPart=1)
        print("starting TS")
        TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)
        # SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c)
    if ucbSim:
        # if justUCB != 'no':
        #     print("RUNNING ADA-ETC HERE w UCB1 and TS and BAI_ETC!!!!")
        #     ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        #     print("starting BAI-ETC")
        #     BAI_ETC = fA.bai_etc(armInstances_, endSim_, K_list_, T_list_)
        #     print("starting TS")
        #     TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)
        naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_, improved=improved, ucbPart=1)

    print("took " + str(time.time() - start_) + " seconds")
    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_, 'm': 1, 'Switch': Switch, 'NADA': NADA}
    _ = None
    if plots:
        a = 0 if ucbSim else 1
        if justUCB == 'no':
            for i in range(a, 2):
                plot_fixed_m(i, K_list_, T_list, naiveUCB1_, TS, ADAETC_, ETC_,
                             NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if ucbSim and (justUCB == 'no'):
            plot_fixed_m(3, K_list_, T_list, naiveUCB1_, TS, ADAETC_, ETC_,
                         NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if Switch == 'yes':
            plot_fixed_m(4, K_list_, T_list, naiveUCB1_, TS, ADAETC_, _, _, _, _, Switch_, params_)
            # plot_fixed_m(5, K_list_, T_list, naiveUCB1_, TS, ADAETC_, _, _, _, _, Switch_, params_)
        if ucbSim and (justUCB == 'yes'):
            plot_fixed_m(4, K_list_, T_list, naiveUCB1_, TS, ADAETC_, _, _, _, _, _, params_, BAI_ETC)
            plot_fixed_m(5, K_list_, T_list, naiveUCB1_, TS, ADAETC_, _, _, _, _, _, params_, BAI_ETC)


def mGeneral(K_list_, T_list_, numArmDists_, endSim_, m_, alpha__, numOpt_, delt_, improved=True, NADA='no'):
    print("Running m =", m_)
    start_ = time.time()
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__,
                                                   numOpt_, delt_, verbose=True)
        print(str(numOpt_) + ' optimal arms')

    m_NADAETC_ = None
    m_naiveUCB1 = fA.m_naiveUCB1(armInstances_, endSim_, K_list_, T_list_, m_, improved=improved, ucbPart=1)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, endSim, K_list_, T_list_, m_)
    RADAETC_ = fA.RADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, endSim_, K_list_, T_list_, m_)
    # if NADA == 'yes':
    #     m_NADAETC_ = fA.m_NADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, m_, improved=improved, ucbPart=1)
    print("took " + str(time.time() - start_) + " seconds")

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': 4, 'delta': delt_, 'm': m_, 'NADA': NADA, 'Switch': 'no'}

    # first argument is set to 2 to use the general m plots
    plot_fixed_m(2, K_list_, T_list, m_naiveUCB1, None, m_ADAETC_, m_ETC_, m_NADAETC_, m_UCB1_stopping_, RADAETC_,
                 None, params_)


if __name__ == '__main__':

    mvals = np.array([1])
    Kvals = np.array([16])
    for i in range(1):
        K_list = np.array([Kvals[i]]) #  np.array([8])  # 8 instead of 10??
        T_list = np.arange(1, 11) * 100  # np.array([15000])
        m = mvals[i]
        numArmDists = 100
        endSim = 50
        doing = 'mEq1'  # 'm1', 'mGeq1', 'm1bar', 'market', 'rott'

        chart('mEq1')
        exit()
        a = 0 #if Kvals[i] == 8 else 1
        for j in range(a, 2):
            alpha_ = (j == 0) * 0 + (j == 1) * 0.4
            print("m ", m, " K ", K_list, " alpha", alpha_)
            if doing == 'm1':
                # fixed mean rewards throughout, m = 1
                mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
                       numOpt_=1, delt_=0, plots=True, ucbSim=True, improved=True, fixed='no', justUCB='no', Switch='no', NADA='no')
                # # fixed='Intervals' or 'Gap' or anything else

            elif doing == 'mGeq1':
                # fixed mean rewards throughout, m > 1
                mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_,
                         numOpt_=1, delt_=0, improved=True, NADA='no')
            elif doing == 'market':
                algs = {'ADA-ETC': {}}#, 'UCB1-I-s': {}, 'ETC': {}}
                numSim = 1
                replicate = 1
                excels = False if numSim > 1 or replicate > 1 else True
                figs = False  # do you want the figures or not
                correction = True
                totalWorkers = 500
                T = 200
                K = 20
                m = 5
                totalC = 100  # int(totalWorkers / m)  # total cohorts
                totalPeriods = totalC * T
                workerArrProb = (K / T)
                roomForError = 1  # need to pay attention to this, algs skip if they have less than m budget left for pulls
                alpha = 0
                excludeZeros = False  # if want to graph rewards w.r.t. cohort graduations rather than time
                cumulative = False  # if  want to graph rewards as a running total
                exploreLess = True  # if true makes all \tau values T/K^(2/3)

                results, normalizedResults, rewards, normalizedRewards = {}, {}, {}, {}
                results2, normalizedResults2, rewards2, normalizedRewards2 = {}, {}, {}, {}
                for keys in algs.keys():
                    results[keys] = {}
                    results[keys]['reward'] = np.zeros(m)
                    results[keys]['stError'] = np.zeros(m)
                    normalizedResults[keys] = {}
                    normalizedResults[keys]['reward'] = np.zeros(m)
                    normalizedResults[keys]['stError'] = np.zeros(m)
                    rewards[keys] = np.zeros((numSim, m))
                    normalizedRewards[keys] = np.zeros((numSim, m))

                    results2[keys] = {}
                    results2[keys]['reward'] = np.zeros(m)
                    results2[keys]['stError'] = np.zeros(m)
                    normalizedResults2[keys] = {}
                    normalizedResults2[keys]['reward'] = np.zeros(m)
                    normalizedResults2[keys]['stError'] = np.zeros(m)
                    rewards2[keys] = np.zeros((numSim, m))
                    normalizedRewards2[keys] = np.zeros((numSim, m))

                start = time.time()
                for sim in range(numSim):
                    print(datetime.now().time())
                    print("\nSim " + str(sim + 1) + " out of " + str(numSim) + "\n")
                    # generate all arms
                    numAllArms = int(int(workerArrProb * totalPeriods * 1.2) / K)  # room for extra arms in case more shows up
                    # generate the arms, single row contains the arms that will be in a single cohort
                    print("An instance sample", end=" ")
                    armsGenerated = gA.generateArms(K_list=[K], T_list=[1], numArmDists=numAllArms, alpha=alpha)
                    for i in range(m):
                        workerArr = np.random.binomial(1, (np.ones(totalPeriods) * workerArrProb))  # worker arrival stream
                        m_cohort = i + 1
                        makesACohort = int(m_cohort * int(K / m))
                        rewardGrouping = 200  # (m_cohort / m) * T * 2
                        res = DynamicMarketSim(algs=algs, m=m, K=K, T=T, m_cohort=m_cohort,
                                               totalCohorts=totalC, workerArrival=workerArr,
                                               armsGenerated=armsGenerated.reshape(int(K * numAllArms / makesACohort),
                                                                                   makesACohort),
                                               exploreLess=True, rewardGrouping=rewardGrouping,
                                               excludeZeros=excludeZeros, cumulative=cumulative,
                                               excels=excels, replicate=replicate, correction=False)
                        res2 = DynamicMarketSim(algs=algs, m=m, K=K, T=T, m_cohort=m_cohort,
                                                totalCohorts=totalC, workerArrival=workerArr,
                                                armsGenerated=armsGenerated.reshape(int(K * numAllArms / makesACohort),
                                                                                    makesACohort),
                                                exploreLess=True, rewardGrouping=rewardGrouping,
                                                excludeZeros=excludeZeros, cumulative=cumulative,
                                                excels=excels, replicate=replicate, correction=True)

                        for keys in algs.keys():
                            rewards[keys][sim, i] = res['rews'][keys]
                            normalizedRewards[keys][sim, i] = res['rewsNormalized'][keys]
                            rewards2[keys][sim, i] = res2['rews'][keys]
                            normalizedRewards2[keys][sim, i] = res2['rewsNormalized'][keys]

                for keys in algs.keys():
                    for i in range(m):
                        results[keys]['reward'][i] = np.mean(rewards[keys][:, i])
                        results[keys]['stError'][i] = np.sqrt(np.var(rewards[keys][:, i] / numSim))
                        normalizedResults[keys]['reward'][i] = np.mean(normalizedRewards[keys][:, i])
                        normalizedResults[keys]['stError'][i] = np.sqrt(np.var(normalizedRewards[keys][:, i] / numSim))
                        results2[keys]['reward'][i] = np.mean(rewards2[keys][:, i])
                        results2[keys]['stError'][i] = np.sqrt(np.var(rewards2[keys][:, i] / numSim))
                        normalizedResults2[keys]['reward'][i] = np.mean(normalizedRewards2[keys][:, i])
                        normalizedResults2[keys]['stError'][i] = np.sqrt(np.var(normalizedRewards2[keys][:, i] / numSim))


                print("\nTook " + str(time.time() - start) + " seconds")
                print("=" * 40)
                for keys in algs.keys():
                    print(keys, results[keys])
                    print(keys, normalizedResults[keys])
                    print(keys, "normalize", results2[keys])
                    print(keys, "normalize", normalizedResults2[keys])
                # plot_market_mBased(algs, m, K, T, results, totalC, numSim, replicate, correction, exploreLess)
                # plot_market_mBased(algs, m, K, T, normalizedResults, totalC, numSim, replicate, correction, exploreLess, normalized=True)
                # plot_market_mBased(algs, m, K, T, results2, totalC, numSim, replicate, True)
                # plot_market_mBased(algs, m, K, T, normalizedResults2, totalC, numSim, replicate, True, normalized=True)

            elif doing == 'm1bar':
                # fixed means but difference is specified between two best arms, varying difference between means, m = 1
                # endSim should be 1, numArmDists should be multiple
                mEqOne_barPlots(K_list, T_list, endSim, alpha_,
                                numOpt_=1, generateIns_=numArmDists, rng=11, ucbSim=False, justUCB='no')
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
