import numpy as np


def naiveUCB1(armInstances, startSim, endSim, K_list, T_list, pw):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    bestreward = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    col = -2 * K
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        col += 2 * K
        for a in range(numInstance):
            arms = armInstances[a, col:(col + 2 * K)]
            fullreward = np.zeros(K)
            get_fullreward = True

            for j in range(endSim - startSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)
                for i in range(T):
                    if i < K:
                        pull = i
                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rew = np.random.binomial(1, meanreward, 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])
                    else:
                        pull = np.argmax(index)
                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rew = np.random.binomial(1, meanreward, 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])
                    if get_fullreward:
                        for k in range(K):
                            fullreward[k] += arms[k] * np.exp(-arms[k + K] * np.power(i, pw))

                reward_sim[a] += sum(cumulative_reward)
                bestreward[t] = max(fullreward)
                get_fullreward = False

                regret_sim[a] += bestreward[t] - max(cumulative_reward)
                subOptRewards_sim[a] += max(pulls) / T
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

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
    print("Best reward is ", bestreward)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}


def ADAETC(armInstances, startSim, endSim, K_list, T_list, pw):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    bestreward = np.zeros(numT)
    col = -2 * K
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        col += 2 * K

        for a in range(numInstance):
            arms = armInstances[a, col:(col + 2 * K)]
            fullreward = np.zeros(K)
            get_fullreward = True

            for j in range(endSim - startSim):
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
                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rew = np.random.binomial(1, meanreward, 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          2 * np.sqrt(max(np.log(T / (K * np.power(pulls[pull], 3 / 2))), 0)
                                                      / pulls[pull]) * (pullEach > pulls[pull])
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])
                    else:
                        pull = np.argmax(indexhigh)
                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rew = np.random.binomial(1, meanreward, 1)
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

                    if get_fullreward:
                        for k in range(K):
                            fullreward[k] += arms[k] * np.exp(-arms[k + K] * np.power(i, pw))
                            stoppedTime = i + 1

                for i in range(T - int(sum(pulls)), T):
                    meanreward = arms[pull_arm] * np.exp(-arms[pull_arm + K] * np.power(pulls[pull_arm], pw))
                    cumulative_reward[pull_arm] += np.random.binomial(1, meanreward, 1)
                pulls[pull_arm] += int(T - sum(pulls))
                if get_fullreward:
                    for i in range(stoppedTime, T):
                        for k in range(K):
                            fullreward[k] += arms[k] * np.exp(-arms[k + K] * np.power(i, pw))

                largestPull = pulls[pull_arm]
                interim = [aa for ii, aa in enumerate(pulls) if aa < largestPull]
                secondLargestPull = max(interim) if len(interim) > 0 else largestPull

                reward_sim[a] += sum(cumulative_reward)
                bestreward[t] = max(fullreward)
                get_fullreward = False

                regret_sim[a] += bestreward[t] - cumulative_reward[pull_arm]
                subOptRewards_sim[a] += (largestPull / max(secondLargestPull, 1))
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

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
    print("Best reward is ", bestreward)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}


def Rotting(armInstances, startSim, endSim, K_list, T_list, pw, sigma, deltaZero, alpha):
    def filterRotting(armIndices, armRewards, windowHere, deltaHere, sigmaHere):
        # armIndices keeps the indices of arms, armRewards keeps the relevant mean reward due to those arms
        candidateArms = []
        confidence = np.sqrt(2 * sigmaHere * sigmaHere / windowHere * np.log(1 / deltaHere))
        highMu = max(armRewards)
        for k in range(len(armIndices)):
            deltaVal = highMu - armRewards[k]
            if deltaVal <= 2 * confidence:
                candidateArms.append(armIndices[k])

        return candidateArms

    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    bestreward = np.zeros(numT)
    col = -2 * K
    for t in range(numT):
        T = T_list[t]
        print("Rotting, T ", T)
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        col += 2 * K

        for a in range(numInstance):
            arms = armInstances[a, col:(col + 2 * K)]
            fullreward = np.zeros(K)
            get_fullreward = True

            for j in range(endSim - startSim):
                # empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                # index = np.zeros(K)
                cumulative_reward = np.zeros(K)
                rewards = np.zeros((K, T))
                for i in range(T):
                    if i < K:
                        pull = i
                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rewards[pull, int(pulls[pull])] = np.random.binomial(1, meanreward, 1)
                        cumulative_reward[pull] += rewards[pull, int(pulls[pull])]
                        pulls[pull] += 1
                        # empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                    else:
                        delta = deltaZero / (K * np.power(i, alpha))
                        window = 1
                        K_filtering = np.arange(0, K)
                        filtering = True

                        while filtering:
                            rewards_windowed = np.zeros(len(K_filtering))
                            for k in range(len(K_filtering)):
                                arm = K_filtering[k]
                                for h in range(window):
                                    rewards_windowed[k] += rewards[arm, int(pulls[arm] - 1 - h)]
                                rewards_windowed[k] /= window
                            K_filtering = filterRotting(K_filtering, rewards_windowed, window, delta, sigma)
                            sub_pulls = pulls[K_filtering]
                            for k in range(len(K_filtering)):
                                arm = K_filtering[k]
                                if pulls[arm] == window:
                                    filtering = False
                                    pull = K_filtering[np.argmin(sub_pulls)]
                            window += 1

                        meanreward = arms[pull] * np.exp(-arms[pull + K] * np.power(pulls[pull], pw))
                        rewards[pull, int(pulls[pull])] = np.random.binomial(1, meanreward, 1)
                        cumulative_reward[pull] += rewards[pull, int(pulls[pull])]
                        pulls[pull] += 1

                    if get_fullreward:
                        for k in range(K):
                            fullreward[k] += arms[k] * np.exp(-arms[k + K] * np.power(i, pw))

                reward_sim[a] += sum(cumulative_reward)
                bestreward[t] = max(fullreward)
                get_fullreward = False

                regret_sim[a] += bestreward[t] - max(cumulative_reward)
                subOptRewards_sim[a] += max(pulls) / T
            reward_sim[a] /= (endSim - startSim)
            regret_sim[a] /= (endSim - startSim)
            subOptRewards_sim[a] /= (endSim - startSim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

    print("Rotting Bandits results:")
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
    print("Best reward is ", bestreward)
    print()
    return {'regret': regret,
            'standardError': stError,
            'pullRatios': subOptRewards}
