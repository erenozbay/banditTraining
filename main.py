import numpy as np
import pandas as pd
import time


def generateArms(K_list, T_list, numArmDists, alpha):
    ncol = sum(K_list) * len(T_list)
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        armInstances[i, :] = np.random.uniform(alpha, 1 - alpha, ncol)

    return armInstances


def generateTwoFartherArms(T_list, numArmDists):
    ncol = 2 * len(T_list)
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        arms = np.zeros(ncol)
        for j in range(len(T_list)):
            diff = np.random.uniform(0, 1 / np.power(T_list[j], 1 / 3), 1) + 1 / np.power(T_list[j], 1 / 3)
            first = 0.5 - diff
            second = 0.5 + diff
            arms[j * 2] = first
            arms[j * 2 + 1] = second
        armInstances[i, :] = arms

    print(armInstances)
    return armInstances


def generateTwoCloserArms(T_list, numArmDists):
    ncol = 2 * len(T_list)
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        arms = np.zeros(ncol)
        for j in range(len(T_list)):
            diff = np.random.uniform(0, 1 / np.sqrt((T_list[j])), 1) + 1 / np.sqrt((T_list[j]))
            first = 0.5 - diff
            second = 0.5 + diff
            arms[j * 2] = first
            arms[j * 2 + 1] = second
        armInstances[i, :] = arms

    print(armInstances)
    return armInstances


def naiveUCB1(armInstances, startSim, endSim, K_list, T_list):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                np.random.seed(j)

                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)

                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])
                    else:
                        pull = np.argmax(index)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])

                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                subOptRewards_sim[a] += np.abs(cumulative_reward[0] - cumulative_reward[1]) / (max(arms) * T)
                subOptRewardsTot_sim[a] += (cumulative_reward[0] + cumulative_reward[1]) / (max(arms) * T)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            subOptRewardsTot_sim[a] /= (endSim - startSim)

        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("Naive UCB1 results:")
    print("K: " + str(K) + ", and T")
    print(T_list)
    print("Regrets")
    print(regret)
    print("Standard errors")
    print(stError)
    print("Ratio of difference between cumulative rewards to the benchmark")
    print(subOptRewards)
    print("Ratio of total cumulative rewards to the benchmark")
    print(subOptRewardsTot)
    print()
    return regret, stError


def ETC(varyingK, armInstances, startSim, endSim, K_list, T_list):
    if varyingK:  # fix T and vary K values
        T = T_list[0]
        numK = len(K_list)
        numInstance = len(armInstances)

        regret = np.zeros(numK)
        stError = np.zeros(numK)
        for t in range(numK):
            K = K_list[t]
            regret_sim = np.zeros(numInstance)

            for a in range(numInstance):
                firstK = sum(K_list[:(t + 1)]) - K
                lastK = sum(K_list[:(t + 1)])
                arms = armInstances[a, firstK:lastK]

                for j in range(endSim - startSim):
                    np.random.seed(j)
                    empirical_mean = np.zeros(K)
                    pulls = np.zeros(K)
                    indexhigh = np.zeros(K)
                    indexlow = np.zeros(K)
                    cumulative_reward = np.zeros(K)

                    for i in range(T):
                        if i < K:
                            pull = i
    else:  # fix K and vary T values
        K = K_list[0]
        numT = len(T_list)
        numInstance = len(armInstances)

        regret = np.zeros(numT)
        stError = np.zeros(numT)
        subOptRewards = np.zeros(numT)
        subOptRewardsTot = np.zeros(numT)
        for t in range(numT):
            T = T_list[t]
            regret_sim = np.zeros(numInstance)
            subOptRewards_sim = np.zeros(numInstance)
            subOptRewardsTot_sim = np.zeros(numInstance)

            for a in range(numInstance):
                arms = armInstances[a, (t * K):((t + 1) * K)]

                for j in range(endSim - startSim):
                    np.random.seed(j)

                    empirical_mean = np.zeros(K)
                    pulls = np.zeros(K)
                    index = np.zeros(K)
                    cumulative_reward = np.zeros(K)

                    pullEach = int(np.ceil(np.power(T, 2 / 3)))

                    for i in range(K):
                        pull = i
                        cumulative_reward[pull] += sum(np.random.binomial(1, arms[pull], pullEach))
                        empirical_mean[pull] = cumulative_reward[pull] / pullEach

                    pull = np.argmax(empirical_mean)
                    rew = np.random.binomial(1, arms[pull], T - K * pullEach)
                    cumulative_reward[pull] += sum(rew)

                    largestCumRew = max(cumulative_reward)
                    interim = [a for i, a in enumerate(cumulative_reward) if a < largestCumRew]
                    secondLargestCumRew = max(interim)

                    regret_sim[a] += max(arms) * T - max(cumulative_reward)
                    subOptRewards_sim[a] += (largestCumRew - secondLargestCumRew) / (max(arms) * T)
                    subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
                regret_sim[a] /= (endSim - startSim)
                subOptRewards_sim[a] /= (endSim - startSim)
                subOptRewardsTot_sim[a] /= (endSim - startSim)

            regret[t] = np.mean(regret_sim)
            stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
            subOptRewards[t] = np.mean(subOptRewards_sim)
            subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("ETC results:")
    print("K: " + str(K) + ", and T")
    print(T_list)
    print("Regrets")
    print(regret)
    print("Standard errors")
    print(stError)
    print("Ratio of difference between cumulative rewards to the benchmark")
    print(subOptRewards)
    print("Ratio of total cumulative rewards to the benchmark")
    print(subOptRewardsTot)
    print()
    return regret, stError


def ADAETC(varyingK, armInstances, startSim, endSim, K_list, T_list):
    if varyingK:  # fix T and vary K values
        T = T_list[0]
        numK = len(K_list)
        numInstance = len(armInstances)

        regret = np.zeros(numK)
        stError = np.zeros(numK)
        for t in range(numK):
            K = K_list[t]
            regret_sim = np.zeros(numInstance)

            for a in range(numInstance):
                firstK = sum(K_list[:(t + 1)]) - K
                lastK = sum(K_list[:(t + 1)])
                arms = armInstances[a, firstK:lastK]

                for j in range(endSim - startSim):
                    np.random.seed(j)
                    empirical_mean = np.zeros(K)
                    pulls = np.zeros(K)
                    indexhigh = np.zeros(K)
                    indexlow = np.zeros(K)
                    cumulative_reward = np.zeros(K)

                    for i in range(T):
                        if i < K:
                            pull = i
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            # indexhigh[pull] =
                            # indexlow[pull] =
                        else:
                            pull = np.argmax(indexhigh)
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            # indexhigh[pull] =
                            # indexlow[pull] =

                    regret_sim[a] += max(arms) * T - max(cumulative_reward)
                regret_sim[a] /= (endSim - startSim)

            regret[t] = np.mean(regret_sim)
            stError[t] = np.sqrt(np.var(regret_sim) / numInstance)

    else:  # fix K and vary T values
        K = K_list[0]
        numT = len(T_list)
        numInstance = len(armInstances)

        regret = np.zeros(numT)
        stError = np.zeros(numT)
        for t in range(numT):
            T = T_list[t]
            regret_sim = np.zeros(numInstance)

            for a in range(numInstance):
                arms = armInstances[a, (t * K):((t + 1) * K)]

                for j in range(endSim - startSim):
                    np.random.seed(j)

                    empirical_mean = np.zeros(K)
                    pulls = np.zeros(K)
                    indexhigh = np.zeros(K)
                    indexlow = np.zeros(K)
                    cumulative_reward = np.zeros(K)

                    for i in range(T):
                        if i < K:
                            pull = i
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            # indexhigh[pull] =
                            # indexlow[pull] =
                        else:
                            pull = np.argmax(indexhigh)
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            # indexhigh[pull] =
                            # indexlow[pull] =

                    regret_sim[a] += max(arms) * T - max(cumulative_reward)
                regret_sim[a] /= (endSim - startSim)

            regret[t] = np.mean(regret_sim)
            stError[t] = np.sqrt(np.var(regret_sim) / numInstance)

    print("ADAETC results:")
    print("K:")
    print(K_list)
    print("T:")
    print(T_list)
    print("Regrets")
    print(regret)
    print("Standard errors")
    print(stError)
    print()
    return regret, stError


if __name__ == '__main__':
    K_list = np.array([2])
    T_list = np.array([2000, 4000, 6000, 8000, 10000])
    numArmDists = 100
    alpha = 0.48
    startSim = 0
    endSim = 100

    # armInstances = generateArms(K_list, T_list, numArmDists, alpha)
    # armInstances = generateTwoCloserArms(T_list, numArmDists)
    armInstances = generateTwoFartherArms(T_list, numArmDists)

    varyingK = True if len(K_list) > 1 else False

    start = time.time()
    naiveUCB1(armInstances, startSim, endSim, K_list, T_list)
    ETC(varyingK, armInstances, startSim, endSim, K_list, T_list)
    # ADAETC(varyingK, armInstances, startSim, endSim, K_list, T_list)
    print("took " + str(time.time() - start) + " seconds")
