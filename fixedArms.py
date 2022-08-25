import numpy as np
import time

def naiveUCB1(armInstances, startSim, endSim, K_list, T_list, stime):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    switch_stError = np.zeros((4, numT))
    subOptRewards = np.zeros(numT)
    switch = np.zeros((4, numT))
    # subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        print("T ", T, ", duration ", time.time() - stime)
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        switch_sim = np.zeros((4, numInstance))
        # subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                # np.random.seed(j)

                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                pulls_later = np.zeros(K)
                prev_pull = 0
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
                        prev_pull = pull
                    else:
                        pull = np.argmax(index)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        if i > T * 0.75:
                            pulls_later[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])
                        if i <= T / 4:
                            switch_sim[0, a] += (1 - prev_pull == pull)
                        elif i <= T / 2:
                            switch_sim[1, a] += (1 - prev_pull == pull)
                        elif i <= 3 * T / 4:
                            switch_sim[2, a] += (1 - prev_pull == pull)
                        else:
                            switch_sim[3, a] += (1 - prev_pull == pull)
                        prev_pull = pull

                reward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)

                subOptRewards_sim[a] += (np.sort(pulls_later)[-1] / max(1, np.sort(pulls_later)[-2]))
                # subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
            regret_sim[a] /= (endSim - startSim)
            reward_sim[a] /= (endSim - startSim)
            for i in range(4):
                switch_sim[i, a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            # subOptRewardsTot_sim[a] /= (endSim - startSim)

        regret[t] = np.mean(regret_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        for i in range(4):
            switch[i, t] = np.mean(switch_sim[i])
            switch_stError[i, t] = np.sqrt(np.var(switch_sim[i]) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        # subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("Naive UCB1 results:")
    print("K: " + str(K) + ", and T: ", end=" ")
    print(T_list)
    print("Regrets", end=" ")
    print(regret)
    print("Total Cumulative Rewards", end=" ")
    print(reward)
    print("Standard errors", end=" ")
    print(stError)
    print("Ratio of pulls spent on the most pulled and the second most pulled in the last quarter horizon")
    print(subOptRewards)
    print("Number of switches between arms")
    for i in range(4):
        print("Quarter ",  i)
        print(switch[i])
    print("And their standard errors")
    for i in range(4):
        print("Quarter ",  i)
        print(switch_stError[i])
    # print("Ratio of total cumulative rewards to the benchmark")
    # print(subOptRewardsTot)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards,
            'numSwitches': switch,
            'numSwitchErrors': switch_stError}


def ETC(armInstances, startSim, endSim, K_list, T_list):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    # subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        # subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                # np.random.seed(j)

                empirical_mean = np.zeros(K)
                cumulative_reward = np.zeros(K)

                pullEach = int(np.ceil(np.power(T, 2 / 3)))

                for i in range(K):
                    pull = i
                    cumulative_reward[pull] += sum(np.random.binomial(1, arms[pull], pullEach))
                    empirical_mean[pull] = cumulative_reward[pull] / pullEach

                pull = np.argmax(empirical_mean)
                rew = np.random.binomial(1, arms[pull], int(T - K * pullEach))
                cumulative_reward[pull] += sum(rew)

                largestCumRew = max(cumulative_reward)
                interim = [a for i, a in enumerate(cumulative_reward) if a < largestCumRew]
                secondLargestCumRew = max(interim)
                reward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                subOptRewards_sim[a] += (largestCumRew - secondLargestCumRew) / sum(cumulative_reward)
                # subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
            regret_sim[a] /= (endSim - startSim)
            reward_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            # subOptRewardsTot_sim[a] /= (endSim - startSim)

        regret[t] = np.mean(regret_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        # subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("ETC results:")
    print("K: " + str(K) + ", and T: ", end=" ")
    print(T_list)
    print("Regrets", end=" ")
    print(regret)
    print("Total Cumulative Rewards", end=" ")
    print(reward)
    print("Standard errors", end=" ")
    print(stError)
    print("Ratio of difference between two closest highest cumulative rewards to the total cumulative rewards")
    print(subOptRewards)
    # print("Ratio of total cumulative rewards to the benchmark")
    # print(subOptRewardsTot)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}


def ADAETC(armInstances, startSim, endSim, K_list, T_list):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    # subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        # subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                # np.random.seed(j)

                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(T, 2 / 3)))
                pull_arm = 0
                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(max(np.log(T / (K * np.power(pulls[pull], 3 / 2))), 0)
                                                      / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])
                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(max(np.log(T / (K * np.power(pulls[pull], 3 / 2))), 0)
                                                      / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    if indexlow[lcb] > indexhigh[ucb]:
                        pull_arm = lcb
                        break
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
                pulls[pull_arm] += int(T - sum(pulls))

                largestPull = pulls[pull_arm]
                interim = [aa for ii, aa in enumerate(pulls) if aa < largestPull]
                secondLargestPull = max(interim)

                reward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - cumulative_reward[pull_arm]
                subOptRewards_sim[a] += (largestPull / max(secondLargestPull, 1))
                # subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            # subOptRewardsTot_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        # subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("ADAETC results:")
    print("K: " + str(K) + ", and T: ", end=" ")
    print(T_list)
    print("Regrets", end=" ")
    print(regret)
    print("Total Cumulative Rewards", end=" ")
    print(reward)
    print("Standard errors", end=" ")
    print(stError)
    print("Ratio of pulls spent on the most pulled and the second most pulled in the last quarter horizon")
    print(subOptRewards)
    # print("Ratio of total cumulative rewards to the benchmark")
    # print(subOptRewardsTot)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}


def NADAETC(armInstances, startSim, endSim, K_list, T_list):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    # subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        # subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                # np.random.seed(j)

                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(T, 2 / 3)))
                pull_arm = 0
                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])
                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    if indexlow[lcb] > indexhigh[ucb]:
                        pull_arm = lcb
                        break
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))

                largestCumRew = cumulative_reward[pull_arm]
                interim = [a for i, a in enumerate(cumulative_reward) if a < largestCumRew]
                secondLargestCumRew = max(interim)

                reward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - cumulative_reward[pull_arm]
                subOptRewards_sim[a] += (largestCumRew - secondLargestCumRew) / sum(cumulative_reward)
                # subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            # subOptRewardsTot_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        # subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("NADAETC results:")
    print("K: " + str(K) + ", and T: ", end=" ")
    print(T_list)
    print("Regrets", end=" ")
    print(regret)
    print("Total Cumulative Rewards", end=" ")
    print(reward)
    print("Standard errors", end=" ")
    print(stError)
    print("Ratio of difference between two closest highest cumulative rewards to the total cumulative rewards")
    print(subOptRewards)
    # print("Ratio of total cumulative rewards to the benchmark")
    # print(subOptRewardsTot)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}


def UCB1_stopping(armInstances, startSim, endSim, K_list, T_list):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    # subOptRewardsTot = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        # subOptRewardsTot_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim - startSim):
                # np.random.seed(j)

                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(T, 2 / 3)))
                pull_arm = 0
                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - \
                                         2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])
                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - \
                                         2 * np.sqrt(np.log(T) / pulls[pull]) * (pullEach > pulls[pull])

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    if indexlow[lcb] > indexhigh[ucb]:
                        pull_arm = lcb
                        break
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))

                largestCumRew = cumulative_reward[pull_arm]
                interim = [a for i, a in enumerate(cumulative_reward) if a < largestCumRew]
                secondLargestCumRew = max(interim)

                reward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - cumulative_reward[pull_arm]
                subOptRewards_sim[a] += (largestCumRew - secondLargestCumRew) / sum(cumulative_reward)
                # subOptRewardsTot_sim[a] += sum(cumulative_reward) / (max(arms) * T)
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
            # subOptRewardsTot_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        # subOptRewardsTot[t] = np.mean(subOptRewardsTot_sim)

    print("UCB1 with stopping results:")
    print("K: " + str(K) + ", and T: ", end=" ")
    print(T_list)
    print("Regrets", end=" ")
    print(regret)
    print("Total Cumulative Rewards", end=" ")
    print(reward)
    print("Standard errors", end=" ")
    print(stError)
    print("Ratio of difference between two closest highest cumulative rewards to the total cumulative rewards")
    print(subOptRewards)
    # print("Ratio of total cumulative rewards to the benchmark")
    # print(subOptRewardsTot)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}
