import discreteDist_fixedArms as dFA
from supportingMethods import *
import time
from datetime import datetime
import generateArms as gA


def doThis(doing):

    if doing == "amazon":
        start = time.time()
        K_list = np.array([6])  # np.array([8])
        T_list = np.arange(1, 11) * 100  # np.array([100])  # np.array([15000])
        numIns = 100
        elements = np.array([0.2, 0.4, 0.6, 0.8, 1])
        ### ipad case ###
        # probabilities = np.array([[0.07, 0.05, 0.06, 0.12, 0.7],
        #                           [0.06, 0, 0.09, 0.15, 0.7],
        #                           [0.04, 0.04, 0.02, 0.13, 0.77],
        #                           [0.06, 0.02, 0.04, 0.1, 0.78],
        #                           [0, 0.07, 0.04, 0.12, 0.77]])
        # best = np.max(np.dot(probabilities, elements))
        # item = 'ipad'  # total num of ratings across all items is 650, minimum is 60
        ### ipad case ###
        ### car phone holder ###
        # probabilities = np.array([[0.05, 0.02, 0.04, 0.08, 0.81],
        #                           [0.06, 0.03, 0.05, 0.1, 0.76],
        #                           [0.05, 0.03, 0.07, 0.15, 0.7],
        #                           [0.04, 0.06, 0.07, 0.16, 0.67],
        #                           [0.06, 0.03, 0.08, 0.13, 0.7]])
        # best = np.max(np.dot(probabilities, elements))
        # item = 'phoneHolder'  # total num of ratings across all items is 37k, minimum is 800
        ### car phone holder ###
        ### iphone case ###
        # probabilities = np.array([[0.02, 0.02, 0.05, 0.13, 0.78],
        #                           [0.06, 0.05, 0.08, 0.14, 0.67],
        #                           [0.04, 0.03, 0.07, 0.15, 0.71],
        #                           [0.05, 0.03, 0.07, 0.16, 0.69],
        #                           [0.03, 0.02, 0.05, 0.14, 0.76]])
        # best = np.max(np.dot(probabilities, elements))
        # item = 'iphone'  # total num of ratings across all items is 62k, minimum is 5k
        ### iphone case ###
        ### dashcam ###
        probabilities = np.array([[0.08, 0.05, 0.06, 0.19, 0.62],
                                  [0.12, 0.06, 0.08, 0.18, 0.56],
                                  [0.07, 0.02, 0.06, 0.22, 0.63],
                                  [0.07, 0.03, 0.06, 0.14, 0.7],
                                  [0.05, 0.01, 0.06, 0.26, 0.62],
                                  [0.06, 0.05, 0.09, 0.19, 0.61]])
        best = np.max(np.dot(probabilities, elements))
        item = 'dashcam'  # total num of ratings across all items is 3k, minimum is 250
        ### dashcam ###
        ### snow shovel ###
        probabilities = np.array([[0.13, 0.09, 0.04, 0.18, 0.56],
                                  [0.08, 0.08, 0.1, 0.13, 0.61],
                                  [0.03, 0.03, 0.06, 0.18, 0.7],
                                  [0.02, 0.01, 0.02, 0.07, 0.88],
                                  [0.03, 0.03, 0.08, 0.17, 0.69],
                                  [0.13, 0.06, 0.18, 0.18, 0.45]])
        best = np.max(np.dot(probabilities, elements))
        item = 'shovel'  # total num of ratings across all items is 650, minimum is 60
        ### snow shovel ###
        ### leaf blower ###
        # probabilities = np.array([[0.03, 0.02, 0.07, 0.19, 0.69],
        #                           [0.06, 0.06, 0.08, 0.19, 0.61],
        #                           [0.15, 0.07, 0.1, 0.19, 0.49],
        #                           [0.1, 0.04, 0.07, 0.15, 0.64],
        #                           [0.07, 0.03, 0.08, 0.17, 0.65],
        #                           [0.12, 0.07, 0.1, 0.19, 0.52]])
        # best = np.max(np.dot(probabilities, elements))
        # item = 'blower'  # total num of ratings across all items is 650, minimum is 60
        ### leaf blower ###
        ### humidifier ###
        # probabilities = np.array([[0.05, 0.03, 0.05, 0.15, 0.72],
        #                           [0.04, 0.02, 0.05, 0.13, 0.76],
        #                           [0.15, 0.06, 0.08, 0.14, 0.57],
        #                           [0.08, 0.04, 0.08, 0.16, 0.64],
        #                           [0.09, 0.03, 0.05, 0.12, 0.71],
        #                           [0.11, 0.03, 0.07, 0.14, 0.65]])
        # best = np.max(np.dot(probabilities, elements))
        # item = 'humidifier'  # total num of ratings across all items is 650, minimum is 60
        ### humidifier ###

        naiveUCB1_ = None
        TS = dFA.thompson(probabilities, numIns * 2, 1, K_list, T_list, best)
        ADAETC_ = dFA.ADAETC(probabilities, numIns, 1, K_list, T_list, best, ucbPart=2)
        ETC_ = dFA.ETC(probabilities, numIns, 1, K_list, T_list, best)
        UCB1_stopping_ = dFA.UCB1_stopping(probabilities, numIns, 1, K_list, T_list, best, improved=False, ucbPart=1, NADA=True)
        naiveUCB1_ = dFA.naiveUCB1(probabilities, numIns, 1, K_list, T_list, best, improved=False, ucbPart=1)

        params_ = {'numOpt': 1, 'alpha': 1, 'totalSim': item,
                   'numArmDists': numIns, 'm': 1, 'Switch': 'no', 'NADA': 'no'}
        _ = None
        for i in range(2):
            plot_fixed_m(i, K_list, T_list, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, _, params_)
        plot_fixed_m(-1, K_list, T_list, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, _, params_)
        print("took " + str(time.time() - start) + " seconds")
    elif doing == 'market':
        algs = {'NADA-ETC': {}}  # {'ADA-ETC': {}, 'UCB1-s': {}, 'ETC': {}}
        numSim = 10
        replicate = 2
        excels = False if numSim > 1 or replicate > 1 else True
        figs = False  # do you want the figures or not
        totalWorkers = 500
        T = 100
        K = 10
        m = 5
        totalC = 100  # int(totalWorkers / m)  # total cohorts
        totalPeriods = totalC * T
        workerArrProb = (K / T)
        alpha = 0
        excludeZeros = False  # if want to graph rewards w.r.t. cohort graduations rather than time
        cumulative = False  # if  want to graph rewards as a running total
        exploreLess = True  # if true makes all \tau values T/K^(2/3)

        results, normalizedResults, rewards, normalizedRewards = {}, {}, {}, {}
        results2, normalizedResults2, rewards2, normalizedRewards2 = {}, {}, {}, {}
        QAC, QAC2 = {}, {}
        for keys in algs.keys():
            results[keys] = {}
            results[keys]['reward'] = np.zeros(m)
            results[keys]['stError'] = np.zeros(m)
            normalizedResults[keys] = {}
            normalizedResults[keys]['reward'] = np.zeros(m)
            normalizedResults[keys]['stError'] = np.zeros(m)
            rewards[keys] = np.zeros((numSim, m))
            normalizedRewards[keys] = np.zeros((numSim, m))
            QAC[keys] = np.zeros((numSim, totalPeriods + 1))

            results2[keys] = {}
            results2[keys]['reward'] = np.zeros(m)
            results2[keys]['stError'] = np.zeros(m)
            normalizedResults2[keys] = {}
            normalizedResults2[keys]['reward'] = np.zeros(m)
            normalizedResults2[keys]['stError'] = np.zeros(m)
            rewards2[keys] = np.zeros((numSim, m))
            normalizedRewards2[keys] = np.zeros((numSim, m))
            QAC2[keys] = np.zeros((numSim, totalPeriods + 1))

        start = time.time()
        for sim in range(numSim):
            print(datetime.now().time())
            print("\nSim " + str(sim + 1) + " out of " + str(numSim) + "\n")
            # generate all arms
            numAllArms = int(int(workerArrProb * totalPeriods * 1.2) / K)  # room for extra arms in case more shows up
            # generate the arms, single row contains the arms that will be in a single cohort
            print("An instance sample", end=" ")
            armsGenerated = gA.generateArms(K_list=[K], T_list=[1], numArmDists=numAllArms, alpha=alpha)
            for i in range(m):  # for i in range(m):
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
                # res2 = DynamicMarketSim(algs=algs, m=m, K=K, T=T, m_cohort=m_cohort,
                #                         totalCohorts=totalC, workerArrival=workerArr,
                #                         armsGenerated=armsGenerated.reshape(int(K * numAllArms / makesACohort),
                #                                                             makesACohort),
                #                         exploreLess=True, rewardGrouping=rewardGrouping,
                #                         excludeZeros=excludeZeros, cumulative=cumulative,
                #                         excels=excels, replicate=replicate, correction=True)

                for keys in algs.keys():
                    rewards[keys][sim, i] = res['rews'][keys]
                    normalizedRewards[keys][sim, i] = res['rewsNormalized'][keys]
                    # rewards2[keys][sim, i] = res2['rews'][keys]
                    # normalizedRewards2[keys][sim, i] = res2['rewsNormalized'][keys]
                    # QAC[keys][sim, :] = res['QAC'][keys]
                    # QAC2[keys][sim, :] = res2['QAC'][keys]

        for keys in algs.keys():
            for i in range(m):
                results[keys]['reward'][i] = np.mean(rewards[keys][:, i])
                results[keys]['stError'][i] = np.sqrt(np.var(rewards[keys][:, i] / numSim))
                normalizedResults[keys]['reward'][i] = np.mean(normalizedRewards[keys][:, i])
                normalizedResults[keys]['stError'][i] = np.sqrt(np.var(normalizedRewards[keys][:, i] / numSim))
                # results2[keys]['reward'][i] = np.mean(rewards2[keys][:, i])
                # results2[keys]['stError'][i] = np.sqrt(np.var(rewards2[keys][:, i] / numSim))
                # normalizedResults2[keys]['reward'][i] = np.mean(normalizedRewards2[keys][:, i])
                # normalizedResults2[keys]['stError'][i] = np.sqrt(np.var(normalizedRewards2[keys][:, i] / numSim))


        print("\nTook " + str(time.time() - start) + " seconds")
        print("=" * 40)
        for keys in algs.keys():
            print(keys, results[keys])
            print(keys, normalizedResults[keys])
            # print(keys, "normalize", results2[keys])
            # print(keys, "normalize", normalizedResults2[keys])
        # plot_market_mBased(algs, m, K, T, results, totalC, numSim, replicate, correction, exploreLess)
        # plot_market_mBased(algs, m, K, T, normalizedResults, totalC, numSim, replicate, correction, exploreLess, normalized=True)
        # plot_market_mBased(algs, m, K, T, results2, totalC, numSim, replicate, True)
        # plot_market_mBased(algs, m, K, T, normalizedResults2, totalC, numSim, replicate, True, normalized=True)
        #     pd.DataFrame(np.transpose(QAC[keys])).to_csv("marketSim/QAC_noCorrection_" + str(keys) + ".csv", index=False)
        #     pd.DataFrame(np.transpose(QAC2[keys])).to_csv("marketSim/QAC_Correction_" + str(keys) + ".csv", index=False)


def amazon_(withUCB=False):
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

    def algsAndFigs(itemList, figs=1, title_="figs/fig6/"):
        for item in itemList:
            TS = dFA.thompson(probabilities[item], numIns * 2, 1, K_list, T_list, best[item])
            ADAETC_ = dFA.ADAETC(probabilities[item], numIns, 1, K_list, T_list, best[item], ucbPart=2)
            ETC_ = dFA.ETC(probabilities[item], numIns, 1, K_list, T_list, best[item])
            UCB1_stopping_ = dFA.UCB1_stopping(probabilities[item], numIns, 1, K_list, T_list, best[item],
                                               improved=False, ucbPart=1, NADA=True)
            naiveUCB1_ = dFA.naiveUCB1(probabilities[item], numIns, 1, K_list, T_list, best[item],
                                       improved=False, ucbPart=1)

            params_ = {'numOpt': 1, 'alpha': 1, 'totalSim': item,
                       'numArmDists': numIns, 'm': 1, 'NADA': 'no', 'title': title_}

            plot_fixed_m(figs, K_list, T_list, naiveUCB1_, TS, ADAETC_, ETC_, UCB1_stopping_, params_)
            # if figs = 0, then the figure is with UCB1 and TS
            # if figs = 1, then the figure is only with TS
            # if figs = -1, then the figure is without UCB1 and TS

    if withUCB:
        itemList_ = ['shovel']
        algsAndFigs(itemList=itemList_, figs=0, title_="figs/figAmazonUCB/")

    else:
        itemList_ = ['dashcam', 'shovel', 'blower', 'humidifier']
        algsAndFigs(itemList=itemList_)
