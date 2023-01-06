import numpy as np
from datetime import datetime

def bai_etc(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
    switch_stError = np.zeros((4, numT))
    numPull = np.zeros((4, numT))
    switch = np.zeros((4, numT))
    stError_perSim = np.zeros(int(endSim))
    mostPulled = np.zeros(numT)
    subOptRewards = np.zeros(numT)

    for t in range(numT):
        T = T_list[t]
        print("BAI-ETC, T:", T)

        regret_sim = np.zeros(numInstance)
        cumreward_sim = np.zeros(numInstance)
        cumReg_sim = np.zeros(numInstance)
        reward_sim = np.zeros(numInstance)
        mostPulled_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.zeros(K)
                cumulative_reward = np.zeros(K)

                for i in range(2):
                    pull = i
                    rew = np.random.binomial(1, arms[pull], 1)
                    cumulative_reward[pull] += rew
                    pulls[pull] += 1
                    empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                    index = empirical_mean


                while (np.sum(pulls) <= T) \
                        and (np.abs(index[0] - index[1]) < np.sqrt(8 * np.log(T / np.sum(pulls)) / np.sum(pulls))):

                    for i in range(2):
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index = empirical_mean


                pull = np.argmax(empirical_mean)
                rew = np.random.binomial(1, arms[pull], int(T - np.sum(pulls)))
                cumulative_reward[pull] += sum(rew)

                # done with the simulation of an instance
                subOptRewards_sim[a] += (max(pulls) / T)
                mostPulled_sim[a] += np.argmax(pulls)
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)

            subOptRewards_sim[a] /= endSim
            mostPulled_sim[a] /= endSim
            regret_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim

        mostPulled[t] = np.mean(mostPulled_sim)
        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

        if verbose:
            print("BAI ETC results:")
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


def thompson(armInstances, endSim, K_list, T_list, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

    regret = np.zeros(numT)
    cumreward = np.zeros(numT)
    cumReg = np.zeros(numT)
    reward = np.zeros(numT)
    stError = np.zeros(numT)
    stError_cumReg = np.zeros(numT)
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
        mostPulled_sim = np.zeros(numInstance)
        subOptRewards_sim = np.zeros(numInstance)

        for a in range(numInstance):
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                pulls = np.zeros(K)
                index = np.ones(K)
                cumulative_reward = np.zeros(K)

                for i in range(T):
                    if i < K:
                        pull = i
                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] += rew

                    else:
                        sampling = np.random.beta(index, pulls + 2 - index)
                        pull = np.argmax(sampling)

                        rew = np.random.binomial(1, arms[pull], 1)
                        cumulative_reward[pull] += rew
                        pulls[pull] += 1
                        empirical_mean[pull] = cumulative_reward[pull] / pulls[pull]
                        index[pull] += rew  # empirical_mean[pull] + 2 * np.sqrt(np.log(T) / pulls[pull])

                # done with the simulation of an instance
                subOptRewards_sim[a] += (max(pulls) / T)
                mostPulled_sim[a] += np.argmax(pulls)
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)

            subOptRewards_sim[a] /= endSim
            mostPulled_sim[a] /= endSim
            regret_sim[a] /= endSim
            cumreward_sim[a] /= endSim
            cumReg_sim[a] /= endSim
            reward_sim[a] /= endSim

        mostPulled[t] = np.mean(mostPulled_sim)
        regret[t] = np.mean(regret_sim)
        cumreward[t] = np.mean(cumreward_sim)
        cumReg[t] = np.mean(cumReg_sim)
        reward[t] = np.mean(reward_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)

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


def naiveUCB1(armInstances, endSim, K_list, T_list, improved=False, ucbPart=2, verbose=True):
    print("UCB part is ", ucbPart)
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

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
    slower_v_fasterArmUCB = np.zeros(numT)
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
                        index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T) / pulls[pull])
                        if improved:  # , improved=False
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T / pulls[pull]) / pulls[pull])
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # index[pull] = empirical_mean[pull] + np.sqrt(2 * np.log(T / max(1, denom)) / pulls[pull])
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
                        index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T) / pulls[pull])
                        if improved:  # , improved=False
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(T / pulls[pull]) / pulls[pull])
                            # denom = np.sum(
                            #     [np.sum([min(pulls[k], np.sqrt(pulls[k] * pulls[j])) for k in range(K)]) for j in
                            #      range(K)])
                            # index[pull] = empirical_mean[pull] + np.sqrt(2 * max(1, np.log(T / denom)) / pulls[pull])

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
                subOptRewards_sim[a] += (max(pulls) / T)
                mostPulled_sim[a] += np.argmax(pulls)
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
                slower_v_fasterArmUCB_sim[3, a] /= endSim

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


def m_naiveUCB1(armInstances, endSim, K_list, T_list, m, improved=False, ucbPart=2, verbose=True):
    print("UCB part is ", ucbPart)
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
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(capT) / pulls[pull])
                            if improved:  # , improved=False
                                index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull])
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
                            index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(capT) / pulls[pull])
                            if improved:  # , improved=False
                                index[pull] = empirical_mean[pull] + ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull])

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
        print("Cumulative regrets", end=" ")
        print(cumReg)
        print("Best Arm Rewards", end=" ")
        print(reward)

        print()
    return {'reward': reward,
            'cumreward': cumreward,
            'cumReg': cumReg,
            'regret': regret,
            'standardError': stError}


def ETC(armInstances, endSim, K_list, T_list, verbose=True, pullDiv=1):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

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
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                empirical_mean = np.zeros(K)
                cumulative_reward = np.zeros(K)

                pullEach = int(np.ceil(np.power(T / K, 2 / 3)) / pullDiv)

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


def m_ETC(armInstances, endSim, K_list, T_list, m, verbose=True, pullDiv=1):
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

                pullEach = int(np.ceil(np.power(T / (K - m), 2 / 3)) / pullDiv)

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


def ADAETC_sub(arms, K, T, ucbPart=2, RADA=False):
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
            up = ucbPart * np.sqrt(max(np.log(T / (K_inUCB * np.power(pulls[pull], 3 / 2))), 0) / pulls[pull])
            indexhigh[pull] = empirical_mean[pull] + up * (pullEach > pulls[pull])
            indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * (pullEach > pulls[pull]) * (up > 0)
        else:
            pull = np.argmax(indexhigh)
            rew = np.random.binomial(1, arms[pull], 1)
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


def ADAETC(armInstances, endSim, K_list, T_list, ucbPart=2, verbose=True):
    # fix K and vary T values
    K = K_list[0]
    numT = len(T_list)
    numInstance = len(armInstances)

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
            arms = armInstances[a, (t * K):((t + 1) * K)]

            for j in range(endSim):
                res = ADAETC_sub(arms, K, T, ucbPart)
                cumulative_reward = res['cumulative_reward']
                pulls = res['pulls']
                pull_arm = np.argmax(pulls)
                mostPulled_sim[a] += pull_arm

                largestPull = pulls[pull_arm]
                cumreward_sim[a] += sum(cumulative_reward)
                reward_sim[a] += max(cumulative_reward)
                regret_sim[a] += max(arms) * T - max(cumulative_reward)
                stError_perSim[j] = max(arms) * T - max(cumulative_reward)
                cumReg_sim[a] += max(arms) * T - sum(cumulative_reward)
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


def m_ADAETC(armInstances, endSim, K_list, T_list, m, ucbPart=2, verbose=True):
    # fix K and vary T values
    print("UCB part is ", ucbPart)
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
                                              ucbPart * np.sqrt(max(np.log(capT / ((K - m) * np.power(pulls[pull], 3 / 2))), 0)
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
                                              ucbPart * np.sqrt(max(np.log(capT / ((K - m) * np.power(pulls[pull], 3 / 2))), 0)
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


def UCB1_stopping(armInstances, endSim, K_list, T_list, improved=False, ucbPart=2, NADA=False, verbose=True):
    # fix K and vary T values
    print("UCB part is ", ucbPart, " NADA ", NADA)
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
    stError_cumReg = np.zeros(numT)

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
                        rew = np.random.binomial(1, arms[pull], 1)
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
                                                 ucbPart * np.sqrt(np.log(T / pulls[pull]) / np.power(pulls[pull], 1)) * pullBool
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
        stError_cumReg[t] = np.sqrt(np.var(cumReg_sim) / numInstance)
        subOptRewards[t] = np.mean(subOptRewards_sim)
    if verbose:
        print("UCB1 with stopping results:") if not NADA else print("NADA-ETC results:")
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


def m_UCB1_stopping(armInstances, endSim, K_list, T_list, m, improved=False, ucbPart=2, verbose=True, NADA=False):
    print("UCB part is ", ucbPart, " NADA?", NADA)
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
                            if improved:
                                indexhigh[pull] = empirical_mean[pull] + \
                                                  ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull]) * pullBool
                                if NADA:
                                    indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                                else:
                                    indexlow[pull] = empirical_mean[pull] - \
                                                     ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull]) * pullBool
                            else:
                                indexhigh[pull] = empirical_mean[pull] + \
                                                  ucbPart * np.sqrt(np.log(capT) / pulls[pull]) * pullBool
                                if NADA:
                                    indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                                else:
                                    indexlow[pull] = empirical_mean[pull] - \
                                                 ucbPart * np.sqrt(np.log(capT) / pulls[pull]) * pullBool

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
                            if improved:
                                indexhigh[pull] = empirical_mean[pull] + \
                                                  ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull]) * pullBool
                                if NADA:
                                    indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                                else:
                                    indexlow[pull] = empirical_mean[pull] - \
                                                 ucbPart * np.sqrt(np.log(capT / pulls[pull]) / pulls[pull]) * pullBool
                            else:
                                indexhigh[pull] = empirical_mean[pull] + \
                                                  ucbPart * np.sqrt(np.log(capT) / pulls[pull]) * pullBool
                                if NADA:
                                    indexlow[pull] = empirical_mean[pull] - empirical_mean[pull] * pullBool
                                else:
                                    indexlow[pull] = empirical_mean[pull] - \
                                                 ucbPart * np.sqrt(np.log(capT) / pulls[pull]) * pullBool

                    lcb_set = np.argsort(-indexlow)
                    lcb = lcb_set[m - 1]
                    indexhigh_copy = indexhigh.copy()
                    indexhigh_copy[lcb_set[0:m]] = -1  # make sure that correct arms are excluded from max UCB step
                    ucb = np.argsort(-indexhigh_copy)[0]
                    pullset = lcb_set[0:m]
                    if (indexlow[lcb] > indexhigh[ucb]) or ((indexlow[lcb] >= indexhigh[ucb]) and sum(pulls) > K):
                        stopped = i + 1
                        break

                if T - stopped > 0:
                    for b in range(m):
                        pull = pullset[b]
                        rew = sum(np.random.binomial(1, arms[pull], int(T - stopped)))
                        cumulative_reward[pull] += rew
                        pulls[pull] += int(T - stopped)

                reward_sim[a] += np.mean(cumulative_reward[pullset])
                regret_sim[a] += first_m * int(capT / m) - np.mean(cumulative_reward[pullset])

            reward_sim[a] /= endSim
            regret_sim[a] /= endSim

        reward[t] = np.mean(reward_sim)
        regret[t] = np.mean(regret_sim)
        stError[t] = np.sqrt(np.var(regret_sim) / numInstance)
    if verbose:
        print(str(m) + "-UCB1 with stopping results:") if not NADA else print(str(m) + "-NADA-ETC results:")
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

