import time
import generateArms as gA
import fixedArmsRotting as fAR
from supportingMethods import *
from copy import deepcopy
import fixedArms as fA


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
                if len(T_list_raw) == 1:
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
    elif fixed == 'Delta':
        delt = (np.ones(len(T_list_)) + 3) / 20
        # delt = np.array([delt] * len(T_list_))
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
    naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, Switch_ = None, None, None, None, None, None, None
    if justUCB == 'no':
        ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
        NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_)
        SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c)
    if ucbSim:
        if justUCB != 'no':
            print("RUNNING ADA-ETC HERE w UCB1!!!!")
            ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_)
    if Switch == 'yes':
        Switch_ = fA.Switching(armInstances_, endSim, K_list, T_list)

    print("took " + str(time.time() - start_) + " seconds")
    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_, 'm': 1, 'Switch': Switch}
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
            plot_fixed_m(4, K_list_, T_list, naiveUCB1_, ADAETC_, _, _, _, _, Switch_, params_)
            # plot_fixed_m(5, K_list_, T_list, naiveUCB1_, ADAETC_, _, _, _, _, Switch_, params_)
        if ucbSim and (justUCB == 'yes'):
            plot_fixed_m(4, K_list_, T_list, naiveUCB1_, ADAETC_, _, _, _, _, _, params_)
            plot_fixed_m(5, K_list_, T_list, naiveUCB1_, ADAETC_, _, _, _, _, _, params_)


def mGeneral(K_list_, T_list_, numArmDists_, endSim_, m_, alpha__, numOpt_, delt_):
    print("Running m =", m_)
    start_ = time.time()
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=False)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list, numArmDists_, alpha__,
                                                   numOpt_, delt_, verbose=False)
        print(str(numOpt_) + ' optimal arms')

    m_naiveUCB1 = fA.m_naiveUCB1(armInstances_, endSim_, K_list_, T_list_, m_)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, endSim, K_list_, T_list_, m_)
    RADAETC_ = fA.RADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_ETC_ = fA.m_ETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_NADAETC_ = fA.m_NADAETC(armInstances_, endSim_, K_list_, T_list_, m_)
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, m_)
    print("took " + str(time.time() - start_) + " seconds")

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': 4, 'delta': delt_, 'm': m_}

    # first argument is set to 2 to use the general m plots
    plot_fixed_m(2, K_list_, T_list, m_naiveUCB1, m_ADAETC_, m_ETC_, m_NADAETC_, m_UCB1_stopping_, RADAETC_,
                 None, params_)  # last RADAETC_ is for switching bandits


if __name__ == '__main__':
    K_list = np.array([2])
    T_list = np.arange(1, 21) * 2000  # np.array([15000])
    m = 1
    numArmDists = 50
    alpha_ = 0
    ucbPart = 2
    endSim = 50
    doing = 'market'  # 'm1', 'mGeq1', 'm1bar', 'market', 'rott', 'FIGURE_wo_UCB1', 'FIGURE_trendReverse'

    if doing == 'market':
        numSim = 1
        replicate = 1
        prints = False if numSim > 1 or replicate > 1 else True
        totalWorkers = 500
        T = 100
        K = 20
        m = 5
        totalCohorts = int(totalWorkers / m)
        totalPeriods = totalCohorts * T
        workerArrProb = (K / T)
        roomForError = 1  # need to pay attention to this, algs skip if they have less than m budget left for pulls
        alpha = 0
        excludeZeros = False  # if want to graph rewards w.r.t. cohort graduations rather than time
        cumulative = False  # if  want to graph rewards as a running total
        adaetcRews = np.zeros((numSim * replicate, m))
        start = time.time()
        for sim in range(numSim):
            workerArrival = np.random.binomial(1, (np.ones(totalPeriods) * workerArrProb))  # worker arrival stream
            for i in range(m):
                m_cohort = i + 1
                rewardGrouping = 200  # (m_cohort / m) * T * 2
                for j in range(replicate):
                    adaetcRews[sim * replicate + j, i] = DynamicMarketSim(m=m, K=K, T=T, m_cohort=m_cohort,
                                                                          totalCohorts=totalCohorts,
                                                                          workerArrival=workerArrival,
                                                                          roomForError=roomForError, alpha=alpha,
                                                                          rewardGrouping=rewardGrouping,
                                                                          excludeZeros=excludeZeros,
                                                                          cumulative=cumulative, prints=prints)
        print("took " + str(time.time() - start) + " seconds")
        print("=" * 40)
        print(adaetcRews)
    elif doing == 'm1':
        # fixed mean rewards throughout, m = 1
        mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
               numOpt_=1, delt_=0, plots=True, ucbSim=True, fixed='no', justUCB='yes', Switch='no')
        # # fixed='Intervals' or 'Gap' or anything else
    elif doing == 'mGeq1':
        # fixed mean rewards throughout, m > 1
        mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_,
                 numOpt_=1, delt_=0.3)
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
    elif doing == 'FIGURE_trendReverse':
        _ = None
        params_ = {'numOpt': 1, 'alpha': 0, 'totalSim': 50, 'numArmDists': 50, 'c': 4, 'delta': 0, 'm': 1,
                   'Switch': 'no'}
        ADAETC_ = {'regret': [66.828,  106.2216, 136.3332, 165.4292 ,195.8048 ,226.1592, 240.6792, 264.8548,
 281.8952, 311.9076 ,334.6428 ,344.1032, 363.6984, 388.3564,404.3012, 421.2,
 440.7044, 453.3956 ,476.778 , 492.7364],
                   'standardError': [1.00514023 ,1.80751337, 2.55779065, 2.86525309, 3.46341282,3.86734538,
 4.3151423  ,4.70186376 ,5.11417389, 5.15582829, 6.46010412 ,5.60920453,
 5.71656761 ,7.69002027 ,5.8180688 , 6.84203927, 9.01090234 ,9.32652529,
 8.08757101,6.81546917],
                   'cumReg': [24.8208  ,39.18 ,   47.4164 , 58.3364 , 70.992 ,  84.6028 , 85.4448 , 94.4412,
  96.5488 ,112.4764 ,124.0384 ,120.0512 ,124.4952, 137.9924, 141.1644, 148.3908,
 156.5172 ,156.1212 ,169.5284 ,173.434],
                   'standardError_cumReg': [0.981945,   1.72455666, 2.38831286, 2.64040662, 3.26645917, 3.72519263,
 4.16844541, 4.27413252, 4.79630982, 4.77943052, 6.14895219, 5.33168483,
 5.56381554, 7.08071626, 5.54013627, 6.40693852, 8.75157452, 8.96722898,
 7.50045516, 6.71009931]}
        naiveUCB1_ = {'regret': [172.396 ,  325.7124 , 460.9248  ,598.9228 , 745.0724 , 874.0964 , 994.45,
 1155.0204 ,1283.4336, 1386.5948, 1525.6616, 1659.69 ,  1781.8172, 1905.288,
 1994.03  , 2140.6652, 2258.714 , 2405.5532, 2516.6616, 2603.0776],
                   'standardError': [1.47135497 , 2.3015352  , 3.49601362 , 3.47329824 , 5.84900984 , 6.06535688,
  7.33239316 , 7.385346 ,   7.76466057 , 8.13474704, 11.75291539 ,12.27322156,
 13.19930928, 11.85374526, 16.08457784, 15.08858451, 13.17101223 ,13.63331954,
 16.49754868 ,17.89722114],
                   'cumReg': [27.9388 , 40.616 ,  50.3852,  60.2412 , 67.6304,  75.8004,  77.8088 , 89.9576,
  95.1212, 100.646 , 102.2852, 110.9056, 113.0904, 118.2008, 119.3308, 127.7004,
 130.8536, 139.2208, 142.202,  141.8572],
                   'standardError_cumReg': [0.45294253, 0.63387393, 0.91837558, 0.90496115, 1.09652983, 1.24359292,
 1.21855024, 1.42758634, 1.34096472 ,1.583389 ,  1.78306315, 1.46404317,
 1.53706455 ,1.70422214 ,2.04751504 ,2.12269598, 2.04983624, 1.99459277,
 2.04489923 ,2.22414136]}
        plot_fixed_m(4, np.array([2]), np.arange(1, 21) * 2000, naiveUCB1_, ADAETC_, _, _, _, _, _, params_)
        plot_fixed_m(5, np.array([2]), np.arange(1, 21) * 2000, naiveUCB1_, ADAETC_, _, _, _, _, _, params_)


    elif doing == 'FIGURE_wo_UCB1':
        ### plots
        _ = None
        K_list = np.array([2])
        T_list = np.arange(1, 16) * 1000
        params_ = {'numOpt': 1, 'alpha': 0, 'totalSim': 100, 'numArmDists': 250, 'c': 4, 'delta': 0, 'm': 1,
                   'Switch': 'no'}
        ADAETC_ = {'regret': [26.8037232, 33.44507723, 43.50046911, 45.10047318, 57.1204632, 55.37838916, 65.00169276,
                              69.39592924, 74.40442152, 69.4653679, 83.85782629, 78.11440985, 88.50950931, 84.82735255,
                              87.18839996],
                   'standardError': [0.92278586, 1.2346894, 1.83587324, 1.99937277, 2.56684265, 2.67792064,
                                     3.11918679, 3.65190925, 3.80024214, 3.48524324, 4.49293148, 4.23062444,
                                     5.11763915, 4.65193228, 5.07901265]}
        ETC_ = {'regret': [42.9795232, 69.85735723, 91.94206911, 108.53563318, 131.3070232,
                           143.36654916, 160.59165276, 173.49896924, 190.00782152, 196.8732079,
                           206.48134629, 222.57840985, 237.81174931, 249.06431255, 258.03127996],
                'standardError': [0.97634804, 1.42824374, 1.91190119, 2.34718774, 2.49025746, 3.06110312,
                                  3.23102363, 3.71014002, 3.79655545, 4.09356194, 4.86685831, 4.89329218,
                                  5.28261671, 5.05269203, 6.01428141]}
        NADAETC_ = {'regret': [41.1448432, 63.63887723, 80.00022911, 90.59579318, 106.2102232,
                               111.58030916, 127.89881276, 137.12572924, 144.94550152, 140.9242079,
                               156.74622629, 160.52964985, 169.75330931, 173.17055255, 182.86187996],
                    'standardError': [0.92410168, 1.35054562, 1.92528933, 2.31390939, 2.66135955, 3.17184934,
                                      3.51717939, 4.21546905, 4.2179283, 4.32825905, 5.17717659, 5.23373061,
                                      5.91481137, 5.7488579, 6.41058042]}
        UCB1_stopping_ = {
            'regret': [41.1632432, 63.64263723, 80.01630911, 90.47555318, 105.5480632, 111.74402916, 128.01433276,
                       137.23684924, 144.18014152, 140.9373279, 155.86494629, 160.25444985, 169.13266931,
                       173.86719255, 182.46391996],
            'standardError': [0.92020676, 1.3586612, 1.93427622, 2.32327293, 2.66849583, 3.12738024, 3.56647396,
                              4.20222404, 4.25472493, 4.37930785, 5.16894814, 5.17352223, 5.85972487, 5.76336477,
                              6.43846717]}
        SuccElim_ = {'regret': [42.7609632, 68.58419723, 87.60146911, 101.51651318, 118.2227032,
                                127.68318916, 143.77849276, 151.80748924, 165.86826152, 159.3431679,
                                173.68014629, 182.36892985, 190.57458931, 198.01915255, 208.11615996],
                     'standardError': [0.96550269, 1.3978518, 1.85854844, 2.2782777, 2.44918981, 2.99361082,
                                       3.36623452, 3.87447499, 3.95672141, 4.04610312, 4.84178548, 5.00822605,
                                       5.64268555, 5.37960248, 6.21821305]}
        for i in range(1, 2):
            plot_fixed_m(i, K_list, T_list, _, ADAETC_, ETC_,
                         NADAETC_, UCB1_stopping_, SuccElim_, _, params_)

        # alpha 0.4
        params_ = {'numOpt': 1, 'alpha': 0.4, 'totalSim': 100, 'numArmDists': 250, 'c': 4, 'delta': 0, 'm': 1,
                   'Switch': 'no'}
        ADAETC_ = {'regret': [38.73604998, 60.78534215, 76.62593865, 91.79332878, 101.20476967,
                              112.54656458, 123.11224074, 135.88825491, 144.84846409, 154.31550702,
                              157.73783088, 170.05846903, 182.69425039, 189.43844988, 193.95849543],
                   'standardError': [0.37892226, 0.62204727, 0.83054986, 1.1985267, 1.41130222, 1.5594478,
                                     1.89392792, 2.09519289, 2.29140476, 2.49911505, 2.8135394, 3.09023649,
                                     3.27366191, 3.43980388, 3.33733003]}
        ETC_ = {'regret': [42.14180998, 67.09714215, 86.63773865, 103.54672878, 119.03944967,
                           133.47812458, 147.32700074, 160.68009491, 175.35954409, 185.97130702,
                           197.64559088, 208.68994903, 222.92441039, 233.87944988, 238.79821543],
                'standardError': [0.3965223, 0.5735214, 0.69590913, 0.86200728, 1.09389293, 1.08981349,
                                  1.29275083, 1.39622428, 1.40435338, 1.60521799, 1.70410731, 1.82244676,
                                  1.86466615, 2.04862906, 2.14104112]}
        NADAETC_ = {'regret': [42.11920998, 66.99254215, 86.33685865, 103.93444878, 118.54104967,
                               133.47252458, 146.07748074, 161.10625491, 175.93622409, 188.17790702,
                               197.31091088, 208.81026903, 223.79789039, 234.91360988, 239.14677543],
                    'standardError': [0.39134253, 0.56777756, 0.68510201, 0.85853425, 1.08333896, 1.02746662,
                                      1.26498548, 1.46071451, 1.52447829, 1.6292776, 1.77681668, 1.80560368,
                                      1.8918351, 2.04588577, 2.18009702]}
        UCB1_stopping_ = {
            'regret': [42.26752998, 67.45478215, 86.48505865, 104.47568878, 119.09192967,
                       133.00536458, 147.53028074, 160.34261491, 175.46534409, 187.00378702,
                       195.60363088, 210.15926903, 222.77153039, 235.47160988, 239.69121543],
            'standardError': [0.40746235, 0.55436888, 0.70957949, 0.91049031, 1.02389755, 1.08103741,
                              1.30277614, 1.47640943, 1.50815156, 1.59026179, 1.63280718, 1.87538794,
                              2.039537, 2.09318265, 2.16889068]}
        SuccElim_ = {'regret': [42.21564998, 67.51122215, 87.55925865, 104.38728878, 118.68000967,
                                133.77136458, 146.31488074, 161.01977491, 175.19074409, 187.69566702,
                                197.44439088, 207.48954903, 223.23881039, 234.82624988, 240.45905543],
                     'standardError': [0.39577918, 0.56796057, 0.6923111, 0.93204667, 1.0622347, 1.0595299,
                                       1.23784399, 1.42879689, 1.45305207, 1.58206878, 1.63589644, 1.80577838,
                                       2.04134831, 2.08400056, 2.15263996]}
        for i in range(1, 2):
            plot_fixed_m(i, K_list, T_list, _, ADAETC_, ETC_,
                         NADAETC_, UCB1_stopping_, SuccElim_, _, params_)

        ### plots
