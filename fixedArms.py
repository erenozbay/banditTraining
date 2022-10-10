import numpy as np


def naiveUCB1(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    lastTime = np.zeros(numT)
    switch_stError = np.zeros((4, numT))
    numPull = np.zeros((4, numT))
    switch = np.zeros((4, numT))
    stError_perSim = np.zeros(int(endSim))
    slower_v_fasterArmUCB = np.zeros(numT)

    for t in range(numT):
        T = T_list[t]

        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        numPull_sim = np.zeros((4, numInstance))
        switch_sim = np.zeros((4, numInstance))
        lastTime_sim = np.zeros(numInstance)
        # first 2 rows for smallest UCBs up to T/4, next row is for the arm w/ at least T/2 pulls and its index at T/2
        # last row keeps track of the events where the least pulled arm's min UCB beats the other's index at T/2
        slower_v_fasterArmUCB_sim = np.zeros((4, numInstance))

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                pulls_later = np.zeros(K)
                prev_pull = 0
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)
                lastTime_simLocal = 0
                mostPulledArm = 0

                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])
                        prev_pull = pull

                        # policy independent statistics
                        if K == 2:
                            slower_v_fasterArmUCB_sim[pull, a] = index[pull]
                    else:
                        pull = np.argmax(index)

                        # K = 2 special case checks
                        if K == 2:
                            if pulls[0] == pulls[1]:  # denote the last time when the pulls were the same
                                lastTime_simLocal = i + 1
                            if np.abs(index[0] - index[1]) < 1e-8:  # if indices are super close, pick randomly
                                pull = int(np.random.binomial(1, 0.5, 1))
                            if i < T / 25:
                                if index[pull] < slower_v_fasterArmUCB_sim[pull, a]:  # update smallest index so far
                                    slower_v_fasterArmUCB_sim[pull, a] = index[pull]
                            if pulls[pull] == int(T / 2) - 1:
                                slower_v_fasterArmUCB_sim[2, a] = index[pull]
                                mostPulledArm = pull

                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])

                        # policy independent statistics
                        if i > T * 0.5:
                            pulls_later[pull] += 1
                        if i <= T / 4:
                            switch_sim[0, a] += (1 - prev_pull == pull)
                        elif i <= T / 2:
                            switch_sim[1, a] += (1 - prev_pull == pull)
                        elif i <= 3 * T / 4:
                            switch_sim[2, a] += (1 - prev_pull == pull)
                        else:
                            switch_sim[3, a] += (1 - prev_pull == pull)
                        prev_pull = pull

                # done with the simulation of an instance
                lastTime_sim[a] += lastTime_simLocal
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)
                if K > 2:
                    numPull_sim[0, a] += max(pulls) / T
                else:
                    numPull_sim[0, a] += 1 if ((pulls[0] - pulls_later[0]) <= (pulls[1] - pulls_later[1]) and
                                                     (pulls_later[0] > pulls_later[1])) else 0
                    numPull_sim[1, a] += 1 if ((pulls[0] - pulls_later[0]) < (pulls[1] - pulls_later[1]) and
                                                     (pulls_later[0] <= pulls_later[1])) else 0
                    numPull_sim[2, a] += 1 if ((pulls[0] - pulls_later[0]) >= (pulls[1] - pulls_later[1]) and
                                                     (pulls_later[0] < pulls_later[1])) else 0
                    numPull_sim[3, a] += 1 if ((pulls[0] - pulls_later[0]) > (pulls[1] - pulls_later[1]) and
                                                     (pulls_later[0] >= pulls_later[1])) else 0
                    slower_v_fasterArmUCB_sim[3, a] += \
                        slower_v_fasterArmUCB_sim[2, a] < slower_v_fasterArmUCB_sim[int(1 - mostPulledArm), a]

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
                slower_v_fasterArmUCB_sim[3, a] /= endSim

        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        lastTime[t] = np.mean(lastTime_sim)
        for i in range(4):
            switch[i, t] = np.mean(switch_sim[i])
            switch_stError[i, t] = np.sqrt(np.var(switch_sim[i]) / numInstance)
        numPull[0, t] = np.mean(numPull_sim[0, :])
        if K == 2:
            numPull[1, t] = np.mean(numPull_sim[1, :])
            numPull[2, t] = np.mean(numPull_sim[2, :])
            numPull[3, t] = np.mean(numPull_sim[3, :])
            slower_v_fasterArmUCB[t] = np.mean(slower_v_fasterArmUCB_sim[3, :])

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
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(numPull) if K == 2 else print(numPull[0, :])
        print('optimal first two elements; among those, switch first, no switch second') if K == 2 else print()

        print('Number of switches between arms ', end='')
        print(switch[0] + switch[1] + switch[2] + switch[3])

        print("Last time the indices were super close " + str(lastTime) + " stError " +
              str(np.sqrt(np.var(lastTime_sim) / numInstance)))
        print("Frequency of min UCB_{T/25}(least pulled) > UCB(T/2)", end=" ")
        print(slower_v_fasterArmUCB)
        print()
        print(slower_v_fasterArmUCB_sim)
        print("="*50)
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'pullRatios': numPull,
            'numSwitches': switch,
            'numSwitchErrors': switch_stError}


def m_naiveUCB1(armInstances, endSim, K_list, T_list, m, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)

    for t in range(numT):
        capT = T_list[t]
        T = int(capT / m)

        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[:m]])

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pull = 0

                for i in range(T):
                    if i < np.ceil(K / m) and pull < K:
                        for b in range(m):
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(capT) / pulls[pull])
                            pull += 1
                            if pull >= K:
                                break
                    else:
                        pullset = np.argsort(-index)
                        for b in range(m):
                            pull = pullset[b]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            index[pull] = empirical_mean[pull] + 2 * np.sqrt(np.log(capT) / pulls[pull])

                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += np.mean(np.sort(cumulative_reward)[-m:])
                regret_sim[a] += first_m * T - np.mean(np.sort(cumulative_reward)[-m:])
                cumReg_sim[a] += first_m * T - sum(cumulative_reward)

            regret_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim

        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)

    if verbose:
        print(str(m) + "-Naive UCB1 results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Best Arm Rewards", end=" ")
        print(reward)

        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError}


def ETC(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                cumulative_reward = np.zeros(K)

                pullEach = int(np.ceil(np.power(T / K, 2 / 3)))

                for i in range(K):
                    pull = i
                    cumulative_reward[pull] += sum(np.random.binomial(1, arms[pull], pullEach))
                    empirical_mean[pull] = cumulative_reward[pull] / pullEach

                pull = np.argmax(empirical_mean)
                rew = np.random.binomial(1, arms[pull], int(T - K * pullEach))
                cumulative_reward[pull] += sum(rew)

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)
            regret_sim[a] /= endSim
            reward_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim

        regret[t] = np.mean(regret_sim)
        reward[t] = np.mean(reward_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
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
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError}


def m_ETC(armInstances, endSim, K_list, T_list, m, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[:m]])

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                cumulative_reward = np.zeros(K)

                pullEach = int(np.ceil(np.power(T / (K - m), 2 / 3)))

                for i in range(K):
                    pull = i
                    cumulative_reward[pull] += sum(np.random.binomial(1, arms[pull], pullEach))
                    empirical_mean[pull] = cumulative_reward[pull] / pullEach

                pullset = np.argsort(-empirical_mean)[:m]
                for i in range(m):
                    more_pull = int(np.floor((T - K * pullEach) / m))
                    cumulative_reward[pullset[i]] += sum(np.random.binomial(1, arms[pullset[i]], more_pull))

                reward_sim[a] += np.mean(cumulative_reward[pullset])
                regret_sim[a] += first_m * int(T / m) - np.mean(cumulative_reward[pullset])
            regret_sim[a] /= endSim
            reward_sim[a] /= endSim

        regret[t] = np.mean(regret_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
    if verbose:
        print(str(m) + "-ETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arms Rewards", end=" ")
        print(reward)
        print("Standard errors", end=" ")
        print(stError)
        print()
    return {'reward': reward,
            'regret': regret,
            'standardError': stError}


def ADAETC_sub(arms, K, T, RADA=False):
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
            rew = np.random.binomial(1, arms[pull], 1)
            cumulative_reward[pull] += rew
            pulls[pull] += 1
            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
            up = 2 * np.sqrt(max(np.log(T / (K_inUCB * np.power(pulls[pull], 3 / 2))), 0) / pulls[pull])
            indexhigh[pull] = empirical_mean[pull] + up * (pullEach > pulls[pull])
            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull]) * (up > 0)
        else:
            pull = np.argmax(indexhigh)
            rew = np.random.binomial(1, arms[pull], 1)
            cumulative_reward[pull] += rew
            pulls[pull] += 1
            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
            up = 2 * np.sqrt(max(np.log(T / (K_inUCB * np.power(pulls[pull], 3 / 2))), 0) / pulls[pull])
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
    cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
    pulls[pull_arm] += int(T - sum(pulls))

    return {"cumulative_reward": cumulative_reward,
            "pulls": pulls}


def subIndices(K_, m_):
    # subgroup indices, e.g., (K, m): (5, 2) should be [0, 1, 2] & [3, 4];
    # (5, 3) should be [0, 1] & [2, 3] & [4]; (5, 4) should be [0, 1] & [2] & [3] & [4]
    # (7, 3) should be [0, 1, 2] & [3, 4] & [5, 6]
    # larger subsets are earlier
    indices_ = {}
    mParam = np.ceil(K_ / m_)  # 10
    startInd = 0
    for mm in range(m_):  # 3
        endInd = startInd + mParam
        indices_[str(mm)] = np.arange(startInd, endInd)
        startInd = endInd
        mParam -= 1 if K_ - endInd < (m_ - mm - 1) * mParam else 0
    return indices_


def ADAETC(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))
    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                res = ADAETC_sub(arms, K, T)
                cumulative_reward = res['cumulative_reward']
                pulls = res['pulls']
                pull_arm = np.argmax(pulls)

                largestPull = pulls[pull_arm]
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                subOptRewards_sim[a] += (largestPull / T)

            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim
            regret_sim[a] /= endSim
            subOptRewards_sim[a] /= endSim

        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
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
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print("Standard errors per simulation", end=" ")
        print(np.sqrt(np.var(stError_perSim) / endSim))
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'pullRatios': subOptRewards}


def m_ADAETC(armInstances, endSim, K_list, T_list, m, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    for t in range(numT):
        capT = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[0:m]])

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(capT / (K - m), 2 / 3)))
                T = int(capT / m)
                stopped = T
                pull = 0
                pullset = np.empty(K)
                for i in range(T):
                    if i < np.ceil(K / m) and pull < K:
                        for b in range(m):
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              2 * np.sqrt(max(np.log(T / ((K - m) * np.power(pulls[pull], 3 / 2))), 0)
                                                          / pulls[pull]) * (pullEach > pulls[pull])
                            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])
                            pull += 1
                            if pull >= K:
                                break
                    else:
                        pullset = np.argsort(-indexhigh)
                        for b in range(m):
                            pull = pullset[b]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              2 * np.sqrt(max(np.log(T / ((K - m) * np.power(pulls[pull], 3 / 2))), 0)
                                                          / pulls[pull]) * (pullEach > pulls[pull])
                            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull])

                    lcb_set = np.argsort(-indexlow)
                    lcb = lcb_set[m - 1]
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb_set[0:m]] = -1  # make sure that correct arms are excluded from max UCB step
                    ucb = np.argsort(-indexhigh_copy)[0]
                    pullset = lcb_set[0:m]
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        stopped = i + 1
                        break

                if stopped < T:
                    for b in range(m):
                        pull = pullset[b]
                        rew = sum(np.random.binomial(1, arms[pull], int(T - stopped)))
                        cumulative_reward[pull] += rew

                reward_sim[a] += np.mean(cumulative_reward[pullset])
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += first_m * int(capT / m) - np.mean(cumulative_reward[pullset])

            reward_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            regret_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        cumreward[t] = np.mean(cumreward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)

    if verbose:
        print(str(m) + "-ADAETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Best Arms Rewards", end=" ")
        print(reward)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'regret': regret,
            'standardError': stError}


def RADAETC(armInstances, endSim, K_list, T_list, m, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))
    for t in range(numT):
        capT = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[0:m]])
            indices = subIndices(K, m)
            # not controlling for rounding errors in T but grouping K arms properly

            for j in range(endSim):
                # call ADAETC m times with randomly grouped ~K/m arms and T/m pulls per call
                res = {}
                np.random.shuffle(arms)

                for i in range(m):
                    instanceIndices = indices[str(i)].astype(int)
                    subK = len(instanceIndices)
                    armsList = arms[instanceIndices]
                    subT = int(capT / m)
                    res[str(i)] = ADAETC_sub(armsList, subK, subT, RADA=True)

                for i in range(m):
                    cumulative_reward = res[str(i)]['cumulative_reward']
                    reward_sim[a] += max(cumulative_reward) / m
                    cumreward_sim[a] += sum(cumulative_reward) / m
                    regret_sim[a] -= max(cumulative_reward) / m

                regret_sim[a] += first_m * int(capT / m)
                stError_perSim[j] = regret_sim[a]

            reward_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            regret_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        cumreward[t] = np.mean(cumreward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
    if verbose:
        print(str(m) + "-RADAETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arms Rewards", end=" ")
        print(reward)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Standard errors", end=" ")
        print(stError)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'regret': regret,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'standardError': stError}


def NADAETC(armInstances, endSim, K_list, T_list, ucbPart=2, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

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
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                        indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    pull_arm = lcb
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        break
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
                pulls[pull_arm] += int(T - sum(pulls))

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                subOptRewards_sim[a] += (max(pulls) / T)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)

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
        subOptRewards[t] = np.mean(subOptRewards_sim)
    if verbose:
        print("NADAETC results:")
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
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError,
            'maxPulls': subOptRewards}


def m_NADAETC(armInstances, endSim, K_list, T_list, m, ucbPart=2, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    for t in range(numT):
        capT = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[0:m]])

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(capT / (K - m), 2 / 3)))
                T = int(capT / m)
                stopped = T
                pull = 0
                pullset = np.empty(K)
                for i in range(T):
                    if i < np.ceil(K / m) and pull < K:
                        for b in range(m):
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            pullBool = pullEach > pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / pulls[pull]) * pullBool
                            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                            pull += 1
                            if pull >= K:
                                break
                    else:
                        pullset = np.argsort(-indexhigh)
                        for b in range(m):
                            pull = pullset[b]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            pullBool = pullEach > pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / pulls[pull]) * pullBool
                            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool

                    lcb_set = np.argsort(-indexlow)
                    lcb = lcb_set[m - 1]
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb_set[0:m]] = -1  # make sure that correct arms are excluded from max UCB step
                    ucb = np.argsort(-indexhigh_copy)[0]
                    pullset = lcb_set[0:m]
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        stopped = i + 1
                        break

                for i in range(T - stopped):
                    for b in range(m):
                        pull = pullset[b]
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1

                reward_sim[a] += np.mean(cumulative_reward[pullset])
                regret_sim[a] += first_m * int(capT / m) - np.mean(cumulative_reward[pullset])

            reward_sim[a] /= endSim
            regret_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
    if verbose:
        print(str(m) + "-NADAETC results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arms Rewards", end=" ")
        print(reward)
        print("Standard errors", end=" ")
        print(stError)
        print()
    return {'reward': reward,
            'regret': regret,
            'standardError': stError}


def UCB1_stopping(armInstances, endSim, K_list, T_list, ucbPart=2, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

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
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                        lowBool = pullBool
                        indexlow[pull] = empirical_mean[pull] - \
                                         ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * lowBool
                    else:
                        pull = np.argmax(indexhigh)
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        pullBool = pullEach > pulls[pull]
                        indexhigh[pull] = empirical_mean[pull] + \
                                          ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * pullBool
                        lowBool = pullBool
                        indexlow[pull] = empirical_mean[pull] - \
                                         ucbPart * np.sqrt(np.log(T) / np.power(pulls[pull], 1)) * lowBool

                    lcb = np.argmax(indexlow)
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb] = -1
                    ucb = np.argmax(indexhigh_copy)
                    pull_arm = lcb
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        break
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
                pulls[pull_arm] += int(T - sum(pulls))

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                # largestPull = pulls[pull_arm]
                subOptRewards_sim[a] += (max(pulls) / T)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)

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
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError}


def m_UCB1_stopping(armInstances, endSim, K_list, T_list, m, ucbPart=2, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    for t in range(numT):
        capT = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]
            first_m = np.mean(arms[np.argsort(-arms)[0:m]])

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                indexhigh = np.zeros(K)
                indexlow = np.zeros(K)
                cumulative_reward = np.zeros(K)
                pullEach = int(np.ceil(np.power(capT / (K - m), 2 / 3)))
                T = int(capT / m)
                stopped = T
                pull = 0
                pullset = np.empty(K)
                for i in range(T):
                    if i < np.ceil(K / m) and pull < K:
                        for b in range(m):
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            pullBool = pullEach > pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / pulls[pull]) * pullBool
                            lowBool = pullBool
                            indexlow[pull] = empirical_mean[pull] - \
                                             ucbPart * np.sqrt(np.log(T) / pulls[pull]) * lowBool
                            pull += 1
                            if pull >= K:
                                break
                    else:
                        pullset = np.argsort(-indexhigh)
                        for b in range(m):
                            pull = pullset[b]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                            pullBool = pullEach > pulls[pull]
                            indexhigh[pull] = empirical_mean[pull] + \
                                              ucbPart * np.sqrt(np.log(T) / pulls[pull]) * pullBool
                            lowBool = pullBool
                            indexlow[pull] = empirical_mean[pull] - \
                                             ucbPart * np.sqrt(np.log(T) / pulls[pull]) * lowBool

                    lcb_set = np.argsort(-indexlow)
                    lcb = lcb_set[m - 1]
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb_set[0:m]] = -1  # make sure that correct arms are excluded from max UCB step
                    ucb = np.argsort(-indexhigh_copy)[0]
                    pullset = lcb_set[0:m]
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        stopped = i + 1
                        break

                for i in range(T - stopped):
                    for b in range(m):
                        pull = pullset[b]
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1

                reward_sim[a] += np.mean(cumulative_reward[pullset])
                regret_sim[a] += first_m * int(capT / m) - np.mean(cumulative_reward[pullset])

            reward_sim[a] /= endSim
            regret_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
    if verbose:
        print(str(m) + "-UCB1 with stopping results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Standard errors", end=" ")
        print(stError)
        print()
    return {'reward': reward,
            'regret': regret,
            'standardError': stError}


def SuccElim(armInstances, endSim, K_list, T_list, constant_c, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                candidates = np.arange(K)
                cumulative_reward = np.zeros(K)
                delta = np.power(K / T, 1 / 3)
                rounds = int(np.ceil(np.power(T / K, 2 / 3)))

                for i in range(rounds):
                    if i == 0:
                        for ij in range(len(candidates)):
                            pull = candidates[ij]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                    else:
                        # elimination step
                        alpha_t = np.sqrt(np.log(constant_c * K * i * i / delta) / i)
                        highest_mean = max(empirical_mean)
                        confidence = highest_mean - 2 * alpha_t
                        candidates = candidates[empirical_mean[candidates] > confidence]

                        if len(candidates) < 2:
                            break

                        # pull arms
                        for ij in range(len(candidates)):
                            pull = candidates[ij]
                            rew = np.random.binomial(1, arms[pull], 1)
                            cumulative_reward[pull] += rew
                            pulls[pull] += 1
                            empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]

                pull_arm = np.argmax(empirical_mean)
                cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
                pulls[pull_arm] += int(T - sum(pulls))
                largestPull = pulls[pull_arm]

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                subOptRewards_sim[a] += (largestPull / T)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)

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
        subOptRewards[t] = np.mean(subOptRewards_sim)
    if verbose:
        print("Successive elimination results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError}


def Switching(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    reward = np.zeros(numT)
    cumreward = np.zeros(numT)
    stError = np.zeros(numT)
    subOptRewards = np.zeros(numT)
    cumReg = np.zeros(numT)
    stError_perSim = np.zeros(int(endSim))

    for t in range(numT):
        T = T_list[t]
        capT = T  # np.ceil(np.power(K * np.power(T, 2), 1/3))
        regret_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        # find the number of stages
        stage = 0
        sum_stage = np.power(capT, (1 - np.power(1 / 2, stage)))
        while sum_stage < capT:
            stage += 1
            sum_stage += np.power(capT, (1 - np.power(1 / 2, stage)))

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                candidates = np.arange(K)
                cumulative_reward = np.zeros(K)
                delta = 1 / capT if capT == T else np.power(K / T, 1 / 3)
                bb, c1, c2 = 2, 1 / 2, 1

                for i in range(stage):
                    upperC = np.sqrt(c2 * (bb / c1) * (K / np.power(capT, (1 - np.power(1 / 2, i)))) *
                                     np.log(K * stage / delta))
                    pullEachArm = min(np.ceil(np.power(capT, (1 - np.power(1 / 2, i))) / len(candidates)),
                                      (capT - sum(pulls)) / len(candidates))

                    # pull arms
                    for ij in range(len(candidates)):
                        pull = candidates[ij]
                        rew = np.random.binomial(1, arms[pull], int(pullEachArm))
                        cumulative_reward[pull] += sum(rew)
                        pulls[pull] += int(pullEachArm)
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]

                    # elimination step
                    highest_mean = max(empirical_mean)
                    confidence = highest_mean - 2 * upperC
                    candidates = candidates[empirical_mean[candidates] > confidence]

                    if len(candidates) < 2:
                        break

                pull_arm = np.argmax(empirical_mean)
                if int(T - sum(pulls)) > 0:
                    cumulative_reward[pull_arm] += sum(np.random.binomial(1, arms[pull_arm], int(T - sum(pulls))))
                    pulls[pull_arm] += int(T - sum(pulls))

                reward_sim[a] += max(cumulative_reward)
                cumreward_sim[a] += sum(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                subOptRewards_sim[a] += (max(pulls) / T)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)

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
        subOptRewards[t] = np.mean(subOptRewards_sim)
    if verbose:
        print("Switching costs results:")
        print("K: " + str(K) + ", and T: ", end=" ")
        print(T_list)
        print("Regrets", end=" ")
        print(regret)
        print("Standard errors", end=" ")
        print(stError)
        print("Best Arm Rewards", end=" ")
        print(reward)
        print("Total Cumulative Rewards", end=" ")
        print(cumreward)
        print("Ratio of pulls spent on the most pulled arm to horizon T")
        print(subOptRewards)
        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'standardError_perSim': np.sqrt(np.var(stError_perSim) / endSim),
            'regret': regret,
            'standardError': stError}
