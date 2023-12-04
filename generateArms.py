import numpy as np


def generateArms(K_list, T_list, numArmDists, alpha, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        armInstances[i, :] = np.random.uniform(alpha, 1 - alpha, ncol)
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_fixedDelta(K_list, T_list, numArmDists, numOpt, delta, verbose=True):
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
