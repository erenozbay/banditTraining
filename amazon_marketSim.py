import discreteDist_fixedArms as dFA
from supportingMethods import *
import generateArms as gA


def amazon(withUCB=False):
    K_list = np.array([6])
    T_list = np.arange(1, 11) * 100
    numIns = 100
    elements = np.array([0.2, 0.4, 0.6, 0.8, 1])
    probabilities = {}
    best = {}

    # dashcam ###
    probabilities["dashcam"] = np.array([[0.08, 0.05, 0.06, 0.19, 0.62],
                                         [0.12, 0.06, 0.08, 0.18, 0.56],
                                         [0.07, 0.02, 0.06, 0.22, 0.63],
                                         [0.07, 0.03, 0.06, 0.14, 0.7],
                                         [0.05, 0.01, 0.06, 0.26, 0.62],
                                         [0.06, 0.05, 0.09, 0.19, 0.61]])
    best["dashcam"] = np.max(np.dot(probabilities["dashcam"], elements))
    # dashcam ###

    # snow shovel ###
    probabilities["shovel"] = np.array([[0.13, 0.09, 0.04, 0.18, 0.56],
                                        [0.08, 0.08, 0.1, 0.13, 0.61],
                                        [0.03, 0.03, 0.06, 0.18, 0.7],
                                        [0.02, 0.01, 0.02, 0.07, 0.88],
                                        [0.03, 0.03, 0.08, 0.17, 0.69],
                                        [0.13, 0.06, 0.18, 0.18, 0.45]])
    best["shovel"] = np.max(np.dot(probabilities["shovel"], elements))
    # snow shovel ###

    # leaf blower ###
    probabilities["blower"] = np.array([[0.03, 0.02, 0.07, 0.19, 0.69],
                                        [0.06, 0.06, 0.08, 0.19, 0.61],
                                        [0.15, 0.07, 0.1, 0.19, 0.49],
                                        [0.1, 0.04, 0.07, 0.15, 0.64],
                                        [0.07, 0.03, 0.08, 0.17, 0.65],
                                        [0.12, 0.07, 0.1, 0.19, 0.52]])
    best["blower"] = np.max(np.dot(probabilities["blower"], elements))
    # leaf blower ###

    # humidifier ###
    probabilities["humidifier"] = np.array([[0.05, 0.03, 0.05, 0.15, 0.72],
                                            [0.04, 0.02, 0.05, 0.13, 0.76],
                                            [0.15, 0.06, 0.08, 0.14, 0.57],
                                            [0.08, 0.04, 0.08, 0.16, 0.64],
                                            [0.09, 0.03, 0.05, 0.12, 0.71],
                                            [0.11, 0.03, 0.07, 0.14, 0.65]])
    best["humidifier"] = np.max(np.dot(probabilities["humidifier"], elements))
    # humidifier ###

    def algsAndFigs(itemList, figs=1):
        for item in itemList:
            TS = dFA.thompson(probabilities[item], numIns * 2, 1, K_list, T_list, best[item])
            ADAETC_ = dFA.ADAETC(probabilities[item], numIns, 1, K_list, T_list, best[item], ucbPart=2)
            ETC_ = dFA.ETC(probabilities[item], numIns, 1, K_list, T_list, best[item])
            UCB1_stopping_ = dFA.UCB1_stopping(probabilities[item], numIns, 1, K_list, T_list, best[item],
                                               improved=False, ucbPart=1, NADA=True)
            naiveUCB1_ = dFA.naiveUCB1(probabilities[item], numIns, 1, K_list, T_list, best[item],
                                       improved=False, ucbPart=1)

            params_ = {'numOpt': 1, 'alpha': 1, 'totalSim': item,
                       'numArmDists': numIns, 'm': 1, 'NADA': 'no', 'title': "amazon"}

            plot_fixed_m(figs, K_list, T_list, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, params_)
            # if figs = 0, then the figure is with UCB1 and TS
            # if figs = 1, then the figure is only with TS
            # if figs = -1, then the figure is without UCB1 and TS

    if withUCB:
        itemList_ = ['shovel']
        algsAndFigs(itemList=itemList_, figs=0)

    else:
        itemList_ = ['dashcam', 'shovel', 'blower', 'humidifier']
        algsAndFigs(itemList=itemList_)


def market():
    def sim(m_, K_, T_):
        algs = {'ADA-ETC': {}, 'NADA-ETC': {}, 'ETC': {}}
        numSim = 10
        replicate = 2
        excels = True

        totalC = 100  # int(totalWorkers / m)  # total cohorts
        totalPeriods = totalC * T_
        workerArrProb = (K_ / T_)
        alpha = 0
        excludeZeros = False  # if want to graph rewards w.r.t. cohort graduations rather than time
        exploreLess = True  # if true makes all \tau values T/K^(2/3)

        results, rewards, results2, rewards2 = {}, {}, {}, {}
        QAC, QAC2 = {}, {}
        res, res2 = None, None
        for keys in algs.keys():
            results[keys] = {}
            results[keys]['reward'] = np.zeros(m_)
            results[keys]['stError'] = np.zeros(m_)
            rewards[keys] = np.zeros((numSim, m_))
            QAC[keys] = np.zeros((numSim, totalPeriods + 1))

            results2[keys] = {}
            results2[keys]['reward'] = np.zeros(m_)
            results2[keys]['stError'] = np.zeros(m_)
            rewards2[keys] = np.zeros((numSim, m_))
            QAC2[keys] = np.zeros((numSim, totalPeriods + 1))

        for sim_ in range(numSim):
            print("\nSim " + str(sim_ + 1) + " out of " + str(numSim) + "\n")
            # generate all arms
            numAllArms = int(int(workerArrProb * totalPeriods * 1.2) / K_)  # room for extra arms in case more shows up
            # generate the arms, single row contains the arms that will be in a single cohort
            print("An instance sample", end=" ")
            armsGenerated = gA.generateArms(K_list=[K_], T_list=[1], numArmDists=numAllArms, alpha=alpha)
            for i in range(m_):
                workerArr = np.random.binomial(1, (np.ones(totalPeriods) * workerArrProb))  # worker arrival stream
                m_cohort = i + 1
                makesACohort = int(m_cohort * int(K_ / m_))
                res = DynamicMarketSim(algs=algs, m=m_, K=K_, T=T_, m_cohort=m_cohort, totalCohorts=totalC,
                                       workerArrival=workerArr,
                                       armsGenerated=armsGenerated.reshape(int(K_ * numAllArms / makesACohort),
                                                                           makesACohort),
                                       exploreLess=True, excludeZeros=excludeZeros,
                                       excels=excels, replicate=replicate, correction=False)

                if K_ == 20:
                    res2 = DynamicMarketSim(algs=algs, m=m_, K=K_, T=T_, m_cohort=m_cohort, totalCohorts=totalC,
                                            workerArrival=workerArr,
                                            armsGenerated=armsGenerated.reshape(int(K_ * numAllArms / makesACohort),
                                                                                makesACohort),
                                            exploreLess=True, excludeZeros=excludeZeros,
                                            excels=excels, replicate=replicate, correction=True)

                for keys in algs.keys():
                    rewards[keys][sim_, i] = res['rews'][keys]
                    if K_ == 20:
                        rewards2[keys][sim_, i] = res2['rews'][keys]
                        QAC[keys][sim_, :] = res['QAC'][keys]  # no correction
                        QAC2[keys][sim_, :] = res2['QAC'][keys]  # correction

        for keys in algs.keys():
            for i in range(m_):
                results[keys]['reward'][i] = np.mean(rewards[keys][:, i])
                results[keys]['stError'][i] = np.sqrt(np.var(rewards[keys][:, i] / numSim))
                if K_ == 20:
                    results2[keys]['reward'][i] = np.mean(rewards2[keys][:, i])
                    results2[keys]['stError'][i] = np.sqrt(np.var(rewards2[keys][:, i] / numSim))

        #
        print("=" * 40)
        if K_ == 20:  # this will give the csv's for fig10a (for sample paths)
            for keys in algs.keys():
                pd.DataFrame(np.transpose(QAC[keys])).to_csv("marketSim/QAC_noCorrection_"
                                                             + str(keys) + ".csv", index=False)
                pd.DataFrame(np.transpose(QAC2[keys])).to_csv("marketSim/QAC_Correction_"
                                                              + str(keys) + ".csv", index=False)

            # for m_cohort = 5, K = 20, T = 200; sample path of average cohort queue lengths
            options_ = {'No correction': {}, 'Correction': {}}
            colors = ['red', 'navy']
            labels = list(options_.keys())
            counter = 0
            plt.figure(figsize=(7, 5), dpi=100)
            plt.rc('axes', axisbelow=True)
            plt.grid()
            results_path = {'No correction': QAC['ADA-ETC'][:, 0], 'Correction': QAC2['ADA-ETC'][:, 0]}

            for keys in options_.keys():
                plt.plot(np.arange(1, 20001), results_path[keys], color=colors[counter], label=labels[counter],
                         linewidth=1.8)
                counter += 1
            plt.ylabel('Active Cohorts', fontsize=18)
            plt.xlabel('Time', fontsize=18)
            plt.xticks(np.array([0, 5000, 10000, 15000, 20000]), fontsize=15)
            plt.yticks(fontsize=15)
            title = 'marketSim/activeCohorts_100sim.eps'
            plt.legend(loc="upper left", prop={'size': 14})
            plt.savefig(title, format='eps', bbox_inches='tight')

            #
            # and use results2 here for fig10b
            results_corr = {}
            for keys in options_.keys():
                results_corr[keys] = {}
                results_corr[keys]['reward'] = np.zeros(5)
                results_corr[keys]['stError'] = np.zeros(5)
            results_corr['No correction']['reward'] = results['ADA-ETC']['reward'][0]
            results_corr['No correction']['stError'] = results['ADA-ETC']['stError'][0]
            results_corr['Correction']['reward'] = results2['ADA-ETC']['reward'][0]
            results_corr['Correction']['stError'] = results2['ADA-ETC']['stError'][0]
            plot_market_mBased(options_, m_, K_, T_, results_corr, totalC, numSim, replicate, True, True)

        #
        plot_market_mBased(algs, m_, K_, T_, results, totalC, numSim, replicate, False, exploreLess)

    #
    #
    T = 100
    K = 10
    m = 5
    sim(m, K, T)

    #
    T = 200
    K = 20
    m = 5
    sim(m, K, T)
