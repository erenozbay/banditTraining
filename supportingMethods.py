import numpy as np
import matplotlib.pyplot as plt
import fixedArms as fA


def sim_small_mid_large_m(armMeansArray_, arrayK_, arrayT_, m_, ucbPart_, alg):
    output = {'reward': -1e8, 'regret': 1e8}
    if m_ == 1:
        if alg == 'ada' or alg == 'rada':
            output = fA.ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'nada':
            output = fA.NADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.UCB1_stopping(armMeansArray_, 0, 1, arrayK_, arrayT_, ucbPart_,
                                              orig=True, verbose=False)
        elif alg == 'ucb1':
            output = fA.naiveUCB1(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'etc':
            output = fA.ETC(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
    else:
        if alg == 'ada':
            output = fA.m_ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'rada':
            output = fA.RADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'nada':
            output = fA.m_NADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.m_UCB1_stopping(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, ucbPart_,
                                                  orig=True, verbose=False)
        elif alg == 'ucb1':
            output = fA.m_naiveUCB1(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'etc':
            output = fA.m_ETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
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


def plot_fixed_m(i, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, params_):
    numOpt_, alpha__, totSims_, ucbPart_ = params_['numOpt'], params_['alpha'], params_['totalSim'], params_['ucbPart']
    numArmDists_, constant_c, delt_, m_ = params_['numArmDists'], params_['c'], params_['delta'], params_['m']

    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    if i == 0:  # with UCB
        plt.plot(T_list, naiveUCB1_['regret'], color='b', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['regret'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
    if i < 2:
        plt.plot(T_list, ADAETC_['regret'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['regret'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['regret'], color='g', label='ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='g', fmt='o', markersize=4, capsize=4)
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
        plt.plot(T_list, ETC_['regret'], color='g', label='m-ETC')
        plt.errorbar(T_list, ETC_['regret'], yerr=ETC_['standardError'],
                     color='g', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, NADAETC_['regret'], color='magenta', label='m-NADA-ETC')
        plt.errorbar(T_list, NADAETC_['regret'], yerr=NADAETC_['standardError'],
                     color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['regret'], color='navy', label='m-UCB1-s')
        plt.errorbar(T_list, UCB1_stopping_['regret'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, SuccElim_['regret'], color='blue', label='RADA-ETC')  # THIS IS RADA-ETC for this part
        plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                     color='blue', fmt='o', markersize=4, capsize=4)
    if i == 3:  # cumulative regret plots
        plt.plot(T_list, naiveUCB1_['cumReg'], color='b', label='UCB1')
        plt.errorbar(T_list, naiveUCB1_['cumReg'], yerr=naiveUCB1_['standardError'],
                     color='b', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ADAETC_['cumReg'], color='r', label='ADA-ETC')
        plt.errorbar(T_list, ADAETC_['cumReg'], yerr=ADAETC_['standardError'],
                     color='r', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, ETC_['cumReg'], color='g', label='ETC')
        plt.errorbar(T_list, ETC_['cumReg'], yerr=ETC_['standardError'],
                     color='g', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, NADAETC_['cumReg'], color='magenta', label='NADA-ETC')
        plt.errorbar(T_list, NADAETC_['cumReg'], yerr=NADAETC_['standardError'],
                     color='magenta', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, UCB1_stopping_['cumReg'], color='navy', label='UCB1-s')
        plt.errorbar(T_list, UCB1_stopping_['cumReg'], yerr=UCB1_stopping_['standardError'],
                     color='navy', fmt='o', markersize=4, capsize=4)
        plt.plot(T_list, SuccElim_['cumReg'], color='purple', label='SuccElim (c=' + str(constant_c) + ')')
        plt.errorbar(T_list, SuccElim_['cumReg'], yerr=SuccElim_['standardError'],
                     color='purple', fmt='o', markersize=4, capsize=4)

    plt.xticks(T_list)
    plt.ylabel('Regret', fontsize=15) if i != 3 else plt.ylabel('Cumulative Regret', fontsize=15)
    plt.xlabel('T', fontsize=15)

    plt.legend(loc="upper left")
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

    plt.cla()
