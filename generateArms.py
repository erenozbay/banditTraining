import numpy as np


def generateArms(K_list, T_list, numArmDists, alpha, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        armInstances[i, :] = np.random.uniform(alpha, 1 - alpha, ncol)
    if verbose:
        print(armInstances[0])
    return armInstances


def generateRottingArms(K, T_list, numArmDists, alpha, beta):
    ncol = int(K * len(T_list))
    armInstances = np.zeros((numArmDists, ncol * 2))

    for i in range(numArmDists):
        col = 0
        for t in range(len(T_list)):
            armInstances[i, col:(col + K)] = np.random.uniform(alpha, 1 - alpha, K)
            armInstances[i, (col + K):(col + 2 * K)] = np.random.uniform(beta, 2 * beta, K)
            col += 2 * K
    print(armInstances[0])
    return armInstances


def generateArms_fixedDelta(K_list, T_list, numArmDists, alpha, numOpt, delta, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))
    K = K_list[0]

    for i in range(numArmDists):
        col = 0
        for t in range(len(T_list)):
            if K - numOpt - 1 >= 0:
                arms = np.concatenate((np.ones(K - numOpt) * (0.5 - delta[t]), np.ones(numOpt) * 0.5))
                # arms = np.concatenate((np.random.uniform(alpha, 0.5, K - numOpt - 1),
                #                        np.array([0.5]), np.ones(numOpt) * (0.5 + delta[t])))
                if K > 2:
                    np.random.shuffle(arms)  # not shuffling if there are two arms, for experiments where that matters
            else:
                arms = np.ones(numOpt) * (0.5 + delta[t])
            armInstances[i, col:(col + K)] = arms
            col += K
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_fixedGap(K_list, T_list, numArmDists, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))
    K = K_list[0]

    for i in range(numArmDists):
        col = 0
        for t in range(len(T_list)):
            arms = np.arange(1, K + 2) / (K + 1)
            arms = arms[:K]
            np.random.shuffle(arms)
            armInstances[i, col:(col + K)] = arms
            col += K
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_fixedIntervals(K_list, T_list, numArmDists, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))
    K = K_list[0]

    for i in range(numArmDists):
        col = 0
        for t in range(len(T_list)):
            arms = np.arange(K + 1) / K
            for j in range(len(arms) - 1):
                arms[j] = np.random.uniform(arms[j], arms[j + 1], 1)
            arms = arms[:K]
            np.random.shuffle(arms)
            armInstances[i, col:(col + K)] = arms
            col += K
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_marketSim(K_list_, T_list_, totalPeriods_, alpha_, numOptPerPeriod):
    armInstances_ = {}

    if numOptPerPeriod == 0:
        col_s = 0
        allArmInstances_ = np.zeros(int(sum(K_list_)))
        for p in range(totalPeriods_):
            armInstances_[str(p)] = generateArms(K_list=np.array([K_list_[p]]),
                                                 T_list=np.array([T_list_[p]]), numArmDists=1,
                                                 alpha=alpha_, verbose=False)
            allArmInstances_[col_s:(col_s + int(K_list_[p]))] = np.array(armInstances_[str(p)])
            col_s += int(K_list_[p])
    else:
        num = numOptPerPeriod
        allArmInstances_ = generateArms(K_list=np.array([sum(K_list_)]), T_list=np.array([sum(T_list_)]),
                                        numArmDists=1, alpha=alpha_, verbose=False)
        # get the top (totalPeriods_ * numOptPerPeriod)-many arms, put them aside and shuffle the remaining arms
        allArmInstances_ = np.sort(allArmInstances_[0])
        top_numOpt = allArmInstances_[-int(num * totalPeriods_):]
        np.random.shuffle(top_numOpt)
        allArmInstances_ = allArmInstances_[:(int(sum(K_list_)) - int(num * totalPeriods_))]
        np.random.shuffle(allArmInstances_)
        col_s = 0
        for p in range(totalPeriods_):
            armInstances_[str(p)] = np.concatenate((top_numOpt[(p * num):((p + 1) * num)],
                                                    allArmInstances_[col_s:(col_s + int(K_list_[p]) - num)]),
                                                   axis=None)
            col_s += int(K_list_[p]) - num

    return {'arms': armInstances_}
