import numpy as np
import matplotlib.pyplot as plt
import fixedArms as fA


# calls ADA-ETC or m-ADA=ETC depending on the m_ value, for the market simulation
def sim_small_mid_large_m(armMeansArray_, arrayK_, arrayT_, m_, alg):
    reward = -1e8
    regret = 1e8
    if m_ == 1:
        if alg == 'ADAETC':
            ADAETC_ = fA.ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
            reward = ADAETC_['reward']
            regret = ADAETC_['regret']
        elif alg == 'NADAETC':
            NADAETC_ = fA.NADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, verbose=False)
            reward = NADAETC_['reward']
            regret = NADAETC_['regret']
    else:
        if alg == 'ADAETC':
            m_ADAETC_ = fA.m_ADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
            reward = m_ADAETC_['reward']
            regret = m_ADAETC_['regret']
        elif alg == 'NADAETC':
            m_NADAETC_ = fA.m_NADAETC(armMeansArray_, 0, 1, arrayK_, arrayT_, m_, verbose=False)
            reward = m_NADAETC_['reward']
            regret = m_NADAETC_['regret']
    return {'reward': reward,
            'regret': regret}


def init_res():
    res_ = {'UCB1': {}, 'ADAETC': {}, 'ETC': {}, 'NADAETC': {}, 'UCB1-s': {}, 'SuccElim': {}, 'SuccElim6': {}}

    for i in res_.keys():
        res_[i]['Regret'] = []
        res_[i]['Reward'] = []
        res_[i]['cumrew'] = []
        res_[i]['standardError'] = []
        res_[i]['cumReg'] = []

    return res_


def store_res(res_, generateIns__, dif, inputDict_, key_):
    res_[key_]['regret_' + str(dif)] = inputDict_['regret']
    res_[key_]['Regret'].append(inputDict_['regret'][0])
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
    succ_elim6 = res_['SuccElim6']

    length = len(naive_ucb1[title]) if title != 'cumReg' else len(naive_ucb1[title]) - 1
    bar1 = np.arange(length)
    bar2 = [x + bw for x in bar1]
    bar3 = [x + bw for x in bar2]
    bar4 = [x + bw for x in bar3]
    bar5 = [x + bw for x in bar4]
    # bar6 = [x + bw for x in bar5]
    bar7 = [x + bw for x in bar5]
    plt.figure(figsize=(12, 8), dpi=150)

    # print(nadaetc[title])
    # print(nadaetc['standardError'])

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
    # plt.bar(bar6, succ_elim6[title], color='violet', width=bw, edgecolor='grey', label='SuccElim - c(6)')
    if UCBin_:
        plt.bar(bar7, naive_ucb1[title][-length:], yerr=naive_ucb1['standardError'][-length:], color='b',
                width=bw, edgecolor='grey', label='UCB1')

    chartTitle = ''
    if title == 'cumrew':
        chartTitle = 'Cumulative Reward'
        # plt.ylim(ymax=70)
    if title == 'cumReg':
        chartTitle = 'Sum Objective Regret'
        # plt.ylim(ymax=70)
    elif title == 'Reward':
        chartTitle = 'Best Arm Reward'
        # plt.ylim(ymax=40)
    elif title == 'Regret':
        chartTitle = 'Max Objective Regret'
        # plt.ylim(ymax=92)
    plt.ylabel(chartTitle, fontsize=15)
    plt.xlabel(r'$\Delta$', fontweight='bold', fontsize=15)
    plt.xticks([x + bw for x in bar1], delt_[-length:])

    plt.legend(loc="upper left")
    # plt.legend(loc="upper right") if title == 'Regret' else plt.legend(loc="upper left")
    plt.savefig('res/' + str(K_) + 'arms_halfHalfDelta_' + str(numOpt__) + 'optArms_' + title + '_' + str(numSim) +
                'sims_T' + str(T_) + '_' + str(generateIns__) + 'inst_' + str(alp) + 'alpha_UCB' +
                str(UCBin_) + '.eps', format='eps', bbox_inches='tight')
    # plt.show()
    plt.cla()


def plot_fixed_m(i, K_list_, T_list, naiveUCB1_, ADAETC_, ETC_, NADAETC_, UCB1_stopping_, SuccElim_, params_):
    numOpt_ = params_['numOpt']
    alpha__ = params_['alpha']
    totSims_ = params_['totalSim']  # endSim_ - startSim_
    numArmDists_ = params_['numArmDists']
    constant_c = params_['c']
    delt_ = params_['delta']
    m_ = params_['m']
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
    if i == 2:
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
        plt.plot(T_list, SuccElim_['regret'], color='blue', label='RADA-ETC')  # THIS IS RADA-ETC
        plt.errorbar(T_list, SuccElim_['regret'], yerr=SuccElim_['standardError'],
                     color='blue', fmt='o', markersize=4, capsize=4)

    plt.xticks(T_list)
    # plt.ylim(ymax=30)
    plt.ylabel('Regret', fontsize=15)
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
    # plt.show()
    plt.cla()
