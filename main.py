import generateArms as gA
import fixedArmsRotting as fAR
from copy import deepcopy
import fixedArms as fA
from manualFigs import *
from longerPrompts import *

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


def mEqOne_varyGapPlots(K_list_, T_list_, endSim_, alpha__, numOpt_=1, generateIns_=2,
                    rng=11, ucbSim=True, justUCB='no'):
    T_list_raw = deepcopy(T_list_)

    for t in range(len(T_list_raw)):
        T_list_ = np.array([T_list_raw[t]])

        print("Running m = 1, justUCB: " + justUCB)
        start_ = time.time()
        res = init_res()
        delt = np.zeros(rng - 1)

        for i in range(rng - 1):
            multi = 50 if justUCB == 'yes' else 1
            print('Iteration ', i)
            delt[i] = round((1 / ((rng - 1) * 2 * multi)) * (i + 1), 5)
            armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, generateIns_, alpha__, numOpt_,
                                                       delta=np.array([delt[i]]), verbose=True)

            if ucbSim or (justUCB == 'yes'):
                naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_, improved=False, ucbPart=1)
                res = store_res(res, generateIns_, i, naiveUCB1_, 'UCB1')
                TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)
                res = store_res(res, generateIns_, i, TS, 'TS')
            if justUCB == 'no':
                ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_, ucbPart=2)
                res = store_res(res, generateIns_, i, ADAETC_, 'ADAETC')
                ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
                res = store_res(res, generateIns_, i, ETC_, 'ETC')
                # NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
                # res = store_res(res, generateIns_, i, NADAETC_, 'NADAETC')
                UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, improved=False, ucbPart=1)
                res = store_res(res, generateIns_, i, UCB1_stopping_, 'UCB1-s')
                # SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c=4)
                # res = store_res(res, generateIns_, i, SuccElim_, 'SuccElim')
        print("took " + str(time.time() - start_) + " seconds")

        params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'numSim': endSim_,
                   'numArmDists': generateIns_}


        plot_varying_delta(K_list_[0], T_list_[0], res, delt, params_, title='Regret')
        plot_varying_delta(K_list_[0], T_list_[0], res, delt, params_, title='cumReg')


def mEqOne(K_list_, T_list_, numArmDists_, endSim_, alpha__, numOpt_, delt_,
           plots=True, ucbSim=True, improved=False, fixed='whatever', justUCB='no', Switch='no', NADA='yes', fn=1):
    print("Running m = 1, justUCB: " + justUCB + ", Switch: " + Switch + ", NADA: " + NADA, " improved?", improved)
    constant_c = 4
    start_ = time.time()
    if fixed == 'Gap':
        armInstances_ = gA.generateArms_fixedGap(K_list_, T_list_, numArmDists_, verbose=True)  # deterministic
        print("Fixed gaps")
    elif fixed == 'Intervals':
        armInstances_ = gA.generateArms_fixedIntervals(K_list_, T_list_, numArmDists_, verbose=True)  # randomized
        print("Fixed intervals")
    elif fixed == 'Delta':
        delt = (np.ones(len(T_list_)) + 3) / 20
        # delt = np.array([delt] * len(T_list_))
        # can make alpha = 0.5 here to get all suboptimal arms at 0.5 and the single optimal arm at 0.5 + delta
        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, alpha__, numOpt_,
                                                   delta=delt, verbose=True)
    elif justUCB == 'yes':
        if fn == 1:
            def fn(x):
                return (2 / np.power(x, 2 / 5)).round(5)
        elif fn == 2:
            def fn(x):
                return (1 / np.power(x, 1 / 2)).round(5)

        delt = fn(T_list_)
        # print(delt)

        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, alpha__, numOpt_,
                                                   delta=delt, verbose=True)

    else:
        if numOpt_ == 1 and delt_ == 0:
            armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
            print("Single optimal arm with random gap")
        else:
            armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, alpha__,
                                                       numOpt_, delt_, verbose=True)
            if numOpt_ == 1:
                print("Single optimal arm with gap " + str(delt_))
            else:
                print(str(numOpt_) + " opt arms, gap " + str(delt_))
    naiveUCB1_, TS, ADAETC_, ETC_, NADAETC_, BAI_ETC, \
        UCB1_stopping_, SuccElim_, Switch_ = None, None, None, None, None, None, None, None, None
    if justUCB == 'no':
        print("starting TS")
        TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)
        # if NADA == 'yes':
        #     NADAETC_ = fA.NADAETC(armInstances_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, improved=improved, ucbPart=1)
        ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)
        # SuccElim_ = fA.SuccElim(armInstances_, endSim_, K_list_, T_list_, constant_c)
    if ucbSim:
        naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_, improved=improved, ucbPart=1)
        # if justUCB != 'no':
        #     print("RUNNING ADA-ETC HERE w UCB1 and TS and BAI_ETC!!!!")
        #     ADAETC_ = fA.ADAETC(armInstances_, endSim, K_list_, T_list_)
        #     print("starting BAI-ETC")
        #     BAI_ETC = fA.bai_etc(armInstances_, endSim_, K_list_, T_list_)
        #     print("starting TS")
        #     TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)

    print("took " + str(time.time() - start_) + " seconds")
    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_, 'm': 1, 'Switch': Switch, 'NADA': NADA}
    _ = None
    if plots:
        a = 0 if ucbSim else 1
        if justUCB == 'no':
            for i in range(a, 2):
                plot_fixed_m(i, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, ETC_,
                             NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if ucbSim and (justUCB == 'no'):
            plot_fixed_m(3, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, ETC_,
                         NADAETC_, UCB1_stopping_, SuccElim_, Switch_, params_)
        if Switch == 'yes':
            plot_fixed_m(4, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, _, _, Switch_, params_)
            # plot_fixed_m(5, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, _, _, Switch_, params_)
        if ucbSim and (justUCB == 'yes'):
            plot_fixed_m(4, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, _, _, _, params_, BAI_ETC)
            plot_fixed_m(5, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, _, _, _, params_, BAI_ETC)


def mGeneral(K_list_, T_list_, numArmDists_, endSim_, m_, alpha__, numOpt_, delt_, improved=True, NADA='no'):
    print("Running m =", m_)
    start_ = time.time()
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list, T_list_, numArmDists_, alpha__,
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
    plot_fixed_m(2, K_list_, T_list_, m_naiveUCB1, None, m_ADAETC_, m_ETC_, m_NADAETC_, m_UCB1_stopping_, RADAETC_,
                 None, params_)


if __name__ == '__main__':
    K_list = np.array([6]) #  np.array([8])
    T_list = np.arange(1, 11) * 100  # np.array([100])  # np.array([15000])
    m = 1
    alpha_ = 0
    numArmDists = 100
    endSim = 50
    doing = 'amazon'  # 'm1', 'mGeq1', 'm1varyGap', 'market', 'rott', 'amazon'

    if doing == 'amazon':
        doThis(doing)
        exit()
    elif doing == 'market':
        doThis(doing)
        exit()

    # chart('rewardGen')
    # exit()

    print("m ", m, " K ", K_list, " alpha", alpha_)
    if doing == 'm1':
        # fixed mean rewards throughout, m = 1
        mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
               numOpt_=1, delt_=0, plots=True, ucbSim=True, improved=False, fixed='no', justUCB='no', Switch='no', NADA='no')
        # # fixed='Intervals' or 'Gap' or anything else

    elif doing == 'mGeq1':
        # fixed mean rewards throughout, m > 1
        mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_,
                 numOpt_=1, delt_=0, improved=True, NADA='no')

    elif doing == 'm1varyGap':
        # fixed means but difference is specified between two best arms, varying difference between means, m = 1
        # endSim should be 1, numArmDists should be multiple
        mEqOne_varyGapPlots(K_list, T_list, endSim, alpha_,
                        numOpt_=2, generateIns_=numArmDists, rng=101, ucbSim=True, justUCB='no')
        # mEqOne_varyGapPlots(K_list, T_list, endSim, alpha_,
        #                 numOpt_=2, generateIns_=numArmDists, rng=11, ucbSim=True, justUCB='no')
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


    # mvals = np.array([2, 2, 4])
    # Kvals = np.array([4, 8, 8])
    # for i in range(3):
    #     K_list = np.array([Kvals[i]])
    #     m = mvals[i]
    #     T_list = np.arange(1, 11) * 100
    #     numArmDists = 200
    #     endSim = 50
    #     for j in range(2):
    #         alpha_ = 0 * (j == 0) + 0.4 * (j == 1)
    #         print("m ", m, " K ", K_list, " alpha", alpha_)
    #         # mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
    #         #        numOpt_=1, delt_=0, plots=True, ucbSim=True, improved=False, fixed='no', justUCB='no', Switch='no',
    #         #        NADA='no')
    #         mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_, numOpt_=1, delt_=0, improved=False, NADA='no')

    # ucb for large T w/ fixed gaps of order T^a
    # mEqOne(np.array([2]), np.arange(8, 11) * 4000, 50, 50, 0,
    #        numOpt_=1, delt_=0, plots=False, ucbSim=True, improved=False, fixed='no', justUCB='yes', Switch='no',
    #        NADA='no', fn=1)
    # mEqOne(np.array([2]), np.arange(1, 11) * 4000, 50, 50, 0,
    #        numOpt_=1, delt_=0, plots=False, ucbSim=True, improved=False, fixed='no', justUCB='yes', Switch='no',
    #        NADA='no', fn=2)
