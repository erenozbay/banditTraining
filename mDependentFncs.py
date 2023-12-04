import fixedArms as fA
from supportingMethods import *
import generateArms as gA


def mEqOne_varyGapPlots(K_list_, T_list_, endSim_, numOpt_=1, generateIns_=2,
                        rng=11, ucbSim=True, justUCB='no', NADA=False):
    for t in range(len(T_list_)):
        T_list_iter = np.array([T_list_[t]])

        print("Running m = 1, justUCB: " + justUCB)
        res = init_res()
        delt = np.zeros(rng - 1)

        for i in range(rng - 1):
            multi = 50 if justUCB == 'yes' else 1
            print('Iteration', i, '-- optimal arms', numOpt_)
            delt[i] = round((1 / ((rng - 1) * 2 * multi)) * (i + 1), 5)
            armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_iter, generateIns_, numOpt_,
                                                       delta=np.array([delt[i]]), verbose=True)

            if ucbSim or (justUCB == 'yes'):
                naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_iter, improved=False, ucbPart=1)
                res = store_res(res, generateIns_, i, naiveUCB1_, 'UCB1')
                TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_iter)
                res = store_res(res, generateIns_, i, TS, 'TS')

            if justUCB == 'no':
                ADAETC_ = fA.ADAETC(armInstances_, endSim_, K_list_, T_list_iter, ucbPart=2)
                res = store_res(res, generateIns_, i, ADAETC_, 'ADAETC')
                ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_iter)
                res = store_res(res, generateIns_, i, ETC_, 'ETC')
                UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_iter, improved=False,
                                                  ucbPart=1,
                                                  NADA=NADA)
                res = store_res(res, generateIns_, i, UCB1_stopping_, 'UCB1-s')

        params_ = {'numOpt': numOpt_, 'numSim': endSim_,
                   'numArmDists': generateIns_, 'NADA': NADA}

        plot_varying_delta(K_list_[0], T_list_iter[0], res, delt, params_, title='Regret')
        plot_varying_delta(K_list_[0], T_list_iter[0], res, delt, params_, title='cumReg')


def mEqOne(K_list_, T_list_, numArmDists_, endSim_, alpha__, numOpt_, delt_,
           plots=True, ucbSim=True, improved=False, fixed='whatever', justUCB='no', NADA=False, NADAucb=1, fn=1):
    print("Running m = 1, justUCB: " + justUCB + ", NADA: ", NADA, " improved?", improved)
    constant_c = 4
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
        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, numOpt_,
                                                   delta=delt, verbose=True)
    elif justUCB == 'yes':
        if fn == 1:
            def fn(x):
                return (2 / np.power(x, 2 / 5)).round(5)
        elif fn == 2:
            def fn(x):
                return (1 / np.power(x, 1 / 2)).round(5)

        delt = fn(T_list_)

        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_, numOpt_,
                                                   delta=delt, verbose=True)

    else:
        if numOpt_ == 1 and delt_ == 0:
            armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
            print("Single optimal arm with random gap")
        else:
            armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_,
                                                       numOpt_, delt_, verbose=True)
            if numOpt_ == 1:
                print("Single optimal arm with gap " + str(delt_))
            else:
                print(str(numOpt_) + " opt arms, gap " + str(delt_))
    naiveUCB1_, TS, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_ = None, None, None, None, None, None, None
    if justUCB == 'no':
        print("starting TS")
        TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)
        UCB1_stopping_ = fA.UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, improved=improved,
                                          ucbPart=NADAucb, NADA=NADA)
        ADAETC_ = fA.ADAETC(armInstances_, endSim_, K_list_, T_list_)
        ETC_ = fA.ETC(armInstances_, endSim_, K_list_, T_list_)

    if ucbSim:
        naiveUCB1_ = fA.naiveUCB1(armInstances_, endSim_, K_list_, T_list_, improved=improved, ucbPart=1)
        if justUCB != 'no':
            ADAETC_ = fA.ADAETC(armInstances_, endSim_, K_list_, T_list_)
            print("starting TS")
            TS = fA.thompson(armInstances_, endSim_, K_list_, T_list_)

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': constant_c, 'delta': delt_,
               'm': 1, 'NADA': NADA, 'title': 'mEqOne'}
    _ = None
    if plots:
        a = 0 if ucbSim else 1
        if justUCB == 'no':
            for i in range(a, 2):
                plot_fixed_m(i, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, params_)
            plot_fixed_m(-1, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, params_)
        if ucbSim and (justUCB == 'no'):
            plot_fixed_m(3, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, params_)
        if ucbSim and (justUCB == 'yes'):
            plot_fixed_m(4, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, params_)
            plot_fixed_m(5, K_list_, T_list_, naiveUCB1_, TS, ADAETC_, _, _, params_)


def mGeneral(K_list_, T_list_, numArmDists_, endSim_, m_, alpha__, numOpt_, delt_,
             improved=True, UCBin=True, NADA=False, NADAucb=1):
    print("Running m =", m_)
    if numOpt_ == 1:
        armInstances_ = gA.generateArms(K_list_, T_list_, numArmDists_, alpha__, verbose=True)
    else:
        armInstances_ = gA.generateArms_fixedDelta(K_list_, T_list_, numArmDists_,
                                                   numOpt_, delt_, verbose=True)
        print(str(numOpt_) + ' optimal arms')

    m_naiveUCB1 = None
    m_UCB1_stopping_ = fA.m_UCB1_stopping(armInstances_, endSim_, K_list_, T_list_, m_, improved=improved,
                                          ucbPart=NADAucb, NADA=NADA)
    if UCBin:
        m_naiveUCB1 = fA.m_naiveUCB1(armInstances_, endSim_, K_list_, T_list_, m_, improved=improved, ucbPart=1)
    m_ADAETC_ = fA.m_ADAETC(armInstances_, endSim_, K_list_, T_list_, m_, ucbPart=2)
    m_ETC_ = fA.m_ETC(armInstances_, endSim_, K_list_, T_list_, m_)

    params_ = {'numOpt': numOpt_, 'alpha': alpha__, 'totalSim': endSim_,
               'numArmDists': numArmDists_, 'c': 4, 'delta': delt_,
               'm': m_, 'NADA': NADA, 'title': 'mGeneral'}

    # first argument is set to 2 to use the general m plots
    plot_fixed_m(2, K_list_, T_list_, m_naiveUCB1, None, m_ADAETC_, m_ETC_, m_UCB1_stopping_, params_)
