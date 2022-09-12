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


def generateTwoArms(T_list, numArmDists, delta):
    ncol = int(2 * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        arms = np.zeros(ncol)
        for j in range(len(T_list)):
            first = 0.5
            second = 0.5 + delta
            arms[j * 2] = first
            arms[j * 2 + 1] = second
        armInstances[i, :] = arms

    print(armInstances[0])
    return armInstances


def generateArms_fixedDelta_old(K_list, T_list, numArmDists, alpha, numOpt, delta, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        armInstances[i, :] = np.concatenate((np.random.uniform(alpha, 0.5, ncol - numOpt - 1),
                                             np.array([0.5]), np.ones(numOpt) * (0.5 + delta)))
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_fixedDelta(K_list, T_list, numArmDists, alpha, numOpt, delta, verbose=True):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))
    K = K_list[0]

    for i in range(numArmDists):
        col = 0
        for t in range(len(T_list)):
            arms = np.concatenate((np.random.uniform(alpha, 0.5, K - numOpt - 1),
                                   np.array([0.5]), np.ones(numOpt) * (0.5 + delta)))
            np.random.shuffle(arms)
            armInstances[i, col:(col + K)] = arms
            col += K
    if verbose:
        print(armInstances[0])
    return armInstances


def generateArms_marketSim(K_list_, T_list_, totalPeriods_, alpha__, oneOptPerPeriod):
    armInstances_ = {}  # need this in the output
    if oneOptPerPeriod:
        allArmInstances_ = generateArms(K_list=np.array([sum(K_list_)]), T_list=np.array([sum(T_list_)]),
                                        numArmDists=1, alpha=alpha__, verbose=False)
        # get the top totalPeriods_-many arms, put them aside in top_m and shuffle the remaining arms
        allArmInstances_ = np.sort(allArmInstances_[0])
        top_m = allArmInstances_[-totalPeriods_:]
        allArmInstances_ = allArmInstances_[:(int(sum(K_list_)) - totalPeriods_)]
        np.random.shuffle(allArmInstances_)
        col_s = 0
        for p in range(totalPeriods_):
            armInstances_[str(p)] = np.concatenate((top_m[p],
                                                    allArmInstances_[col_s:(col_s + int(K_list_[p]) - 1)]),
                                                   axis=None)
            col_s += int(K_list_[p]) - 1
    else:
        col_s = 0
        allArmInstances_ = np.zeros(int(sum(K_list_)))
        for p in range(totalPeriods_):
            armInstances_[str(p)] = generateArms(K_list=np.array([K_list_[p]]),
                                                 T_list=np.array([T_list_[p]]), numArmDists=1,
                                                 alpha=alpha__, verbose=False)
            allArmInstances_[col_s:(col_s + int(K_list_[p]))] = np.array(armInstances_[str(p)])
            col_s += int(K_list_[p])
        allArmInstances_ = np.sort(allArmInstances_)
        top_m = allArmInstances_[-totalPeriods_:]

    # get the average reward of the top m arms in an instance, in this case m = totalPeriods_
    best_top_m_avg_reward = np.mean(top_m)  # need this in the output
    total_T = sum(T_list_) / totalPeriods_  # need this in the output

    return {'armInstances': armInstances_,
            'best_top_m_avg_reward': best_top_m_avg_reward,
            'total_T': total_T}


# this should be made more clear and concise, basically I should be able to fix one arm and generate others with a gap
def generateMultipleArms(K_list, T_list, numArmDists, pw=1 / 3):
    ncol = int(sum(K_list) * len(T_list))
    armInstances = np.zeros((numArmDists, ncol))

    if numArmDists > 1:
        for i in range(numArmDists):
            arms_T = np.zeros(ncol)
            for t in range(len(T_list)):
                arms = np.zeros(sum(K_list))
                for j in range(len(K_list)):
                    K = K_list[j]
                    within = min(0.5, 1 / np.power(T_list[t], pw))
                    sub_arms = np.random.uniform(0.5 - within, 0.5 + within, K - 1)
                    opt_arm = max(sub_arms) + within
                    if opt_arm > 1:
                        opt_arm = 1
                        sub_arms = sub_arms / opt_arm
                    range_first = 0 if j == 0 else sum(K_list[0:j])
                    range_last = range_first + K - 1
                    arms[range_first:range_last] = sub_arms
                    arms[range_last] = opt_arm
                arms_T[(t * sum(K_list)):((t + 1) * sum(K_list))] = arms
            armInstances[i, :] = arms_T
    else:
        arms_T = np.zeros(ncol)
        for t in range(len(T_list)):
            arms = np.zeros(sum(K_list))
            for j in range(len(K_list)):
                K = K_list[j]
                within = min(0.5, 1 / np.power(T_list[t], pw))
                sub_arms = np.random.uniform(0.5 - within, 0.5 + within, K - 1)
                opt_arm = max(sub_arms) + within
                if opt_arm > 1:
                    opt_arm = 1
                    sub_arms = sub_arms / opt_arm
                range_first = 0 if j == 0 else sum(K_list[0:j])
                range_last = range_first + K - 1
                arms[range_first:range_last] = sub_arms
                arms[range_last] = opt_arm
            arms_T[(t * sum(K_list)):((t + 1) * sum(K_list))] = arms
        armInstances[0, :] = arms_T

    print(armInstances[0])
    return armInstances
