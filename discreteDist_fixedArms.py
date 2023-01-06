import numpy as np
from datetime import datetime

def thompson(armInstances, numIns, endSim, K_list, T_list, best, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = numIns

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
    lastTime = np.zeros(numT)
    switch_stError = np.zeros((4, numT))
    numPull = np.zeros((4, numT))
    switch = np.zeros((4, numT))
    stError_perSim = np.zeros(int(endSim))
    mostPulled = np.zeros(numT)
    subOptRewards = np.zeros(numT)

    for t in range(numT):
        T = T_list[t]
        print("TS, T:", T)

        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        numPull_sim = np.zeros((4, numInstance))
        switch_sim = np.zeros((4, numInstance))
        lastTime_sim = np.zeros(numInstance)
        mostPulled_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.ones((K, 5))  # each arm's running rew is in a row
                cumulative_reward = np.zeros(K)
                lastTime_simLocal = 0

                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        # rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull, (int(rew * 5) - 1)] += 1
                    else:
                        sampling = np.zeros(K)
                        for kk in range(K):
                            sampling[kk] = np.dot(np.random.default_rng().dirichlet(index[kk], 1),
                                                  np.array([0.2, 0.4, 0.6, 0.8, 1]))
                        pull = np.argmax(sampling)

                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull, (int(rew * 5) - 1)] += 1


                # done with the simulation of an instance
                subOptRewards_sim[a] += (max(pulls) / T)
                mostPulled_sim[a] += np.argmax(pulls)
                lastTime_sim[a] += lastTime_simLocal
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += best * T - max(cumulative_reward)
                cumReg_sim[a] += best * T - sum(cumulative_reward)
                stError_perSim[j] = best * T - max(cumulative_reward)

            subOptRewards_sim[a] /= endSim
            mostPulled_sim[a] /= endSim
            regret_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim
            lastTime_sim[a] /= endSim
            for i in range(4):
                switch_sim[i, a] /= endSim
            numPull_sim[0, a] /= endSim
            if K == 2:
                numPull_sim[1, a] /= endSim
                numPull_sim[2, a] /= endSim
                numPull_sim[3, a] /= endSim

        mostPulled[t] = np.mean(mostPulled_sim)
        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        lastTime[t] = np.mean(lastTime_sim)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        for i in range(4):
            switch[i, t] = np.mean(switch_sim[i])
            switch_stError[i, t] = np.sqrt(np.var(switch_sim[i]) / numInstance)
        numPull[0, t] = np.mean(numPull_sim[0, :])
        if K == 2:
            numPull[1, t] = np.mean(numPull_sim[1, :])
            numPull[2, t] = np.mean(numPull_sim[2, :])
            numPull[3, t] = np.mean(numPull_sim[3, :])

        if verbose:
            print("Thompson Sampling results:")
            print("K: " + str(K) + ", and T: ", end=" ")
            print(T_list)
            print("Regrets", end=" ")
            print(regret)
            print("Standard errors", end=" ")
            print(stError)
            print("Total Cumulative Rewards", end=" ")
            print(cumreward)
            print("Cumulative regrets", end=" ")
            print(cumReg)
            print("Cumulative Reward Standard errors", end=" ")
            print(stError_cumReg)
            print("Best Arm Rewards", end=" ")
            print(reward)
            print("Most pulled")
            print(mostPulled)
            print("Ratio of pulls spent on the most pulled arm to horizon T")
            print(subOptRewards)
            print(datetime.now().time())
            print("="*50)
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError,
            'standardError_cumReg': stError_cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'pullRatios': numPull,
            'numSwitches': switch,
            'numSwitchErrors': switch_stError}


def naiveUCB1(armInstances, numIns, endSim, K_list, T_list, best, improved=False, ucbPart=2, verbose=True):
    print("UCB part is ", ucbPart)
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = numIns

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
    lastTime = np.zeros(numT)
    switch_stError = np.zeros((4, numT))
    numPull = np.zeros((4, numT))
    switch = np.zeros((4, numT))
    stError_perSim = np.zeros(int(endSim))
    mostPulled = np.zeros(numT)
    subOptRewards = np.zeros(numT)

    for t in range(numT):
        T = T_list[t]
        print("UCB1, T:", T)

        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        numPull_sim = np.zeros((4, numInstance))
        switch_sim = np.zeros((4, numInstance))
        lastTime_sim = np.zeros(numInstance)
        mostPulled_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)
                lastTime_simLocal = 0

                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T) / pulls[pull])
                        if improved:  # , improved=False
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T / pulls[pull]) / pulls[pull])
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # index[pull] = empirical_mean[pull] + np.sqrt(2 * np.log(T / max(1, denom)) / pulls[pull])
                        prev_pull = pull

                    else:
                        pull = np.argmax(index)

                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T) / pulls[pull])
                        if improved:  # , improved=False
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T / pulls[pull]) / pulls[pull])
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # index[pull] = empirical_mean[pull] + np.sqrt(2 * max(1, np.log(T / denom)) / pulls[pull])

                # done with the simulation of an instance
                subOptRewards_sim[a] += (max(pulls) / T)
                mostPulled_sim[a] += np.argmax(pulls)
                lastTime_sim[a] += lastTime_simLocal
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += best * T - max(cumulative_reward)
                cumReg_sim[a] += best * T - sum(cumulative_reward)
                stError_perSim[j] = best * T - max(cumulative_reward)

            subOptRewards_sim[a] /= endSim
            mostPulled_sim[a] /= endSim
            regret_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim
            lastTime_sim[a] /= endSim
            for i in range(4):
                switch_sim[i, a] /= endSim
            numPull_sim[0, a] /= endSim

        mostPulled[t] = np.mean(mostPulled_sim)
        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        lastTime[t] = np.mean(lastTime_sim)
        subOptRewards[t] = np.mean(subOptRewards_sim)
        for i in range(4):
            switch[i, t] = np.mean(switch_sim[i])
            switch_stError[i, t] = np.sqrt(np.var(switch_sim[i]) / numInstance)

    if verbose:
        print("Naive UCB1 results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Cumulative regrets", end=" ")
        print(cumReg)
        print("Cumulative Reward Standard errors", end=" ")
        print(stError_cumReg)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Most pulled")
        print(mostPulled)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print("="*50)
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError,
            'standardError_cumReg': stError_cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'pullRatios': numPull,
            'numSwitches': switch,
            'numSwitchErrors': switch_stError}


def ETC(armInstances, numIns, endSim, K_list, T_list, best, verbose=True, pullDiv=1):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = numIns

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                cumulative_reward = np.zeros(K)

                pullEach = int(np.ceil(np.power(T / K, 2 / 3)) / pullDiv)

                for i in range(K):
                    pull = i
                    cumulative_reward[pull] += sum(np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), pullEach,
                                                                    p=arms[pull]))
                    empirical_mean[pull] = cumulative_reward[pull] / pullEach

                pull = np.argmax(empirical_mean)
                rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), int(T - K * pullEach), p=arms[pull])
                cumulative_reward[pull] += sum(rew)

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += best * T - max(cumulative_reward)
                cumReg_sim[a] += best * T - sum(cumulative_reward)
                stError_perSim[j] = best * T - max(cumulative_reward)
            regret_sim[a] /= endSim
            reward_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim

        regret[t] = np.mean(regret_sim)
        reward[t] = np.mean(reward_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
    if verbose:
        print("ETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Standard errors", end=" ")
        print(stError)
        print("Cumulative regrets", end=" ")
        print(cumReg)
        print("Cumulative Reward Standard errors", end=" ")
        print(stError_cumReg)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError,
            'standardError_cumReg': stError_cumReg}


def ADAETC_sub(arms, K, T, ucbPart, RADA=False):
    empirical_mean = np.zeros(K)
    pulls = np.zeros(K)
    indexhigh = np.zeros(K)
    indexlow = np.zeros(K)
    cumulative_reward = np.zeros(K)
    pullEach = int(np.ceil(np.power(T / K, 2 / 3)))
    K_inUCB = K
    if RADA:
        pullEach = int(np.ceil(np.power(T / (K - 1), 2 / 3)))
        K_inUCB = K - 1
    pull_arm = 0
    for i in range(T):
        if i < K:
            pull = i
            rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
            cumulative_reward[pull] += rew
            pulls[pull] += 1
            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
            up = ucbPart * np.sqrt(max(np.log(T / (K_inUCB * np.power(pulls[pull], 3 / 2))), 0) / pulls[pull])
            indexhigh[pull] = empirical_mean[pull] + up * (pullEach > pulls[pull])
            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull]) * (up > 0)
        else:
            pull = np.argmax(indexhigh)
            rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
            cumulative_reward[pull] += rew
            pulls[pull] += 1
            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
            up = ucbPart * np.sqrt(max(np.log(T / (K_inUCB * np.power(pulls[pull], 3 / 2))), 0) / pulls[pull])
            indexhigh[pull] = empirical_mean[pull] + up * (pullEach > pulls[pull])
            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull]) * (up > 0)

        lcb = np.argmax(indexlow)
        indexhigh_copy = indexhigh.copy()
        indexhigh_copy[lcb] = -1
        ucb = np.argmax(indexhigh_copy)
        if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
            # and (pulls[lcb] >= pullEach) and (pulls[ucb] >= pullEach)):
            pull_arm = lcb
            break
    cumulative_reward[pull_arm] += sum(np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]),
                                                                    int(T - sum(pulls)), p=arms[pull_arm]))
    pulls[pull_arm] += int(T - sum(pulls))

    return {"cumulative_reward": cumulative_reward,
            "pulls": pulls}


def ADAETC(armInstances, numIns, endSim, K_list, T_list, best, ucbPart=2, verbose=True):
    # fix K and vary T values
    print("UCB part is ", ucbPart)
    K = K_list[0]
    numT = len(T_list)
    numInstance = numIns

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    mostPulled = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)
        mostPulled_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances

            for j in range(endSim):
                res = ADAETC_sub(arms, K, T, ucbPart)
                cumulative_reward = res['cumulative_reward']
                pulls = res['pulls']
                pull_arm = np.argmax(pulls)
                mostPulled_sim[a] += pull_arm

                largestPull = pulls[pull_arm]
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += best * T - max(cumulative_reward)
                stError_perSim[j] = best * T - max(cumulative_reward)
                cumReg_sim[a] += best * T - sum(cumulative_reward)
                subOptRewards_sim[a] += (largestPull / T)

            mostPulled_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim
            regret_sim[a] /= endSim
            subOptRewards_sim[a] /= endSim

        mostPulled[t] = np.mean(mostPulled_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

    if verbose:
        print("ADAETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Cumulative regrets", end=" ")
        print(cumReg)
        print("Cumulative Reward Standard errors", end=" ")
        print(stError_cumReg)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print("Standard errors per simulation", end=" ")
        print(np.sqrt(np.var(stError_perSim) / endSim))
        print("Most pulled")
        print(mostPulled)
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError,
            'standardError_cumReg': stError_cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'pullRatios': subOptRewards}


def UCB1_stopping(armInstances, numIns, endSim, K_list, T_list, best, improved=False, ucbPart=2, NADA=False, verbose=True):
    # fix K and vary T values
    print("UCB part is ", ucbPart, " NADA?", NADA)
    K = K_list[0]
    numT = len(T_list)
    numInstance = numIns

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))
    stError_cumReg = np.zeros(numT)

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(T / K, 2 / 3)))
                pull_arm = 0
                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        if improved:
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # indexhigh[pull] = empirical_mean[pull] + \
                            #               np.sqrt(ucbPart * np.log(T / denom) / np.power(pulls[pull], 1)) * pullBool
                            # indexlow[pull] = empirical_mean[pull] - \
                            #                  np.sqrt(ucbPart * np.log(T / denom) / np.power(pulls[pull], 1)) * pullBool
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T / pulls[pull]) / np.power(pulls[pull], 1)) * pullBool
                            if NADA:
                                indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                            else:
                                indexlow[pull] = empirical_mean[pull] - \
                                                 ucbPart * np.sqrt(np.log(T / pulls[pull]) / np.power(pulls[pull], 1)) * pullBool
                        else:
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                            if NADA:
                                indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                            else:
                                indexlow[pull] = empirical_mean[pull] - \
                                             ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool

                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]), 1, p=arms[pull])
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        if improved:
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # indexhigh[pull] = empirical_mean[pull] + \
                            #               np.sqrt(ucbPart * np.log(T / denom) / np.power(pulls[pull], 1)) * pullBool
                            # indexlow[pull] = empirical_mean[pull] - \
                            #                  np.sqrt(ucbPart * np.log(T / denom) / np.power(pulls[pull], 1)) * pullBool
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(
                                np.log(T / pulls[pull]) / np.power(pulls[pull], 1)) * pullBool
                            if NADA:
                                indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                            else:
                                indexlow[pull] = empirical_mean[pull] - \
                                             ucbPart * np.sqrt(
                                np.log(T / pulls[pull]) / np.power(pulls[pull], 1)) * pullBool
                        else:
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                            if NADA:
                                indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                            else:
                                indexlow[pull] = empirical_mean[pull] - \
                                             ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    pull_arm = lcb
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        break
                cumulative_reward[pull_arm] += sum(np.random.choice(np.array([0.2, 0.4, 0.6, 0.8, 1]),
                                                                    int(T - sum(pulls)), p=arms[pull_arm]))
                pulls[pull_arm] += int(T - sum(pulls))

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += best * T - max(cumulative_reward)
                # largestPull = pulls[pull_arm]
                subOptRewards_sim[a] += (max(pulls) / T)
                stError_perSim[j] = best * T - max(cumulative_reward)
                cumReg_sim[a] += best * T - sum(cumulative_reward)

            reward_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            regret_sim[a] /= endSim
            subOptRewards_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
    if verbose:
        print("UCB1 with stopping results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Standard errors", end=" ")
        print(stError)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Cumulative regrets", end=" ")
        print(cumReg)
        print("Cumulative Reward Standard errors", end=" ")
        print(stError_cumReg)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError,
            'standardError_cumReg': stError_cumReg}
