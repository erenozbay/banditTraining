import numpy as np


class CohortGenerate:
    def __init__(self, cohortNum, armList, K, T, m, alg, exploreLess=False):
        self.alg = alg
        self.cohortNum = cohortNum
        self.generated = False
        self.K = K
        self.T = T
        self.m = m
        self.arms = armList
        self.pulls = np.zeros(self.K)
        self.indexhigh = np.zeros(self.K)
        self.indexlow = np.zeros(self.K)
        self.empirical_mean = np.zeros(self.K)
        self.cumulative_reward = np.zeros(self.K)
        self.pullEach = int(np.ceil(np.power(T / (K - m), 2 / 3))) if m > 1 else int(np.ceil(np.power(T / K, 2 / 3)))
        if exploreLess:
            self.pullEach = int(np.ceil(np.power(T / K, 2 / 3)))
        self.exploitPhase = False
        self.stopped = 0
        self.pullset = np.zeros(m)
        self.currentReward = 0


    def returnIndices(self, budget, pullOne=-2, exploiting=False):
        if pullOne == -2:
            if self.alg != 'ETC':
                pullset = np.argsort(-self.indexhigh) if not exploiting else self.pullset
                for b in range(budget):
                    pull = pullset[b]
                    rew = np.random.binomial(1, self.arms[pull], 1)
                    self.currentReward += rew
                    self.cumulative_reward[pull] += rew
                    self.pulls[pull] += 1
                    if not exploiting:
                        self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                        boolie = self.pullEach > self.pulls[pull]
                        if self.alg == 'ADA-ETC':
                            up = 2 * np.sqrt(max(np.log(self.T / (self.K * np.power(self.pulls[pull], 3 / 2))), 0) / self.pulls[pull])
                            self.indexhigh[pull] = self.empirical_mean[pull] + up * boolie
                            self.indexlow[pull] = self.empirical_mean[pull] - self.empirical_mean[pull] * boolie
                        elif self.alg == 'UCB1-s':
                            confidence = 1 * np.sqrt(np.log(self.T) / np.power(self.pulls[pull], 1)) * boolie
                            self.indexhigh[pull] = self.empirical_mean[pull] + confidence
                            self.indexlow[pull] = self.empirical_mean[pull] - confidence
            else:
                pullset = np.argsort(self.pulls) if not exploiting else self.pullset
                for b in range(budget):
                    pull = pullset[b]
                    rew = np.random.binomial(1, self.arms[pull], 1)
                    self.currentReward += rew
                    self.cumulative_reward[pull] += rew
                    self.pulls[pull] += 1
                    if not exploiting:
                        self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                        self.indexhigh[pull] = self.empirical_mean[pull]
                        self.indexlow[pull] = self.empirical_mean[pull]
        else:  # for one arm pulling in the first few periods
            rew = np.random.binomial(1, self.arms[pullOne], 1)
            pull = pullOne
            self.currentReward += rew
            self.cumulative_reward[pull] += rew
            self.pulls[pull] += 1
            self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
            boolie = self.pullEach > self.pulls[pull]
            if self.alg == 'ADA-ETC':
                up = 2 * np.sqrt(
                    max(np.log(self.T / (self.K * np.power(self.pulls[pull], 3 / 2))), 0) / self.pulls[pull])
                self.indexhigh[pull] = self.empirical_mean[pull] + up * boolie
                self.indexlow[pull] = self.empirical_mean[pull] - self.empirical_mean[pull] * boolie
            elif self.alg == 'UCB1-s':
                confidence = 1 * np.sqrt(np.log(self.T) / np.power(self.pulls[pull], 1)) * boolie
                self.indexhigh[pull] = self.empirical_mean[pull] + confidence
                self.indexlow[pull] = self.empirical_mean[pull] - confidence
            elif self.alg == 'ETC':
                self.empirical_mean[pull] = self.cumulative_reward[pull] / self.pulls[pull]
                self.indexhigh[pull] = self.empirical_mean[pull]
                self.indexlow[pull] = self.empirical_mean[pull]

    def exploit(self):
        lcb_set = np.argsort(-self.indexlow)
        lcb = lcb_set[self.m - 1]
        indexhigh_copy = self.indexhigh.copy()
        indexhigh_copy[lcb_set[0:self.m]] = -1  # make sure that correct arms are excluded from max UCB step
        ucb = np.argsort(-indexhigh_copy)[0]
        pullset = lcb_set[0:self.m]
        if self.alg != 'ETC':
            if (self.indexlow[lcb] > self.indexhigh[ucb]) or ((self.indexlow[lcb] >= self.indexhigh[ucb])
                                                              and sum(self.pulls) > self.K):
                self.stopped = int(sum(self.pulls) / self.m)
                self.exploitPhase = True
                self.pullset = pullset
        else:
            if all(self.pulls >= self.pullEach):
                self.stopped = int(sum(self.pulls) / self.m)
                self.exploitPhase = True
                self.pullset = pullset

    def step(self, budget):
        done = False
        self.currentReward = 0
        if not self.exploitPhase:  # do not check if I am in the exploitation phase after I get in it
            if any(self.pulls < 1):
                for i in range(self.K):
                    if self.pulls[i] == 0:
                        budget -= 1
                        self.returnIndices(1, i)
                    if budget <= 0:
                        break
            if budget >= self.m:  # I want to pull m arms at all times, not anything less
                self.returnIndices(self.m)
                self.exploit()   # check if I can exploit the next time

            return {'budget': budget,  # this budget I can use to return to the pool of jobs, maybe?
                    'done': done,
                    'realtime_reward': self.currentReward}  # / self.m}

        else:  # already in the exploitation phase
            if (self.T - sum(self.pulls)) > 0:
                self.returnIndices(min(self.m, self.T - sum(self.pulls)), -2, True)
                # print("Total pulls spent ", sum(self.pulls), " vs T ", self.T)
            else:
                done = True

            return {'budget': budget,
                    'done': done,
                    'final_reward': #self.T / self.m * np.sum(self.arms[np.argsort(-self.arms)[0:self.m]]) -
                                    np.sum(self.cumulative_reward[self.pullset]),
                    # np.mean(self.cumulative_reward[self.pullset]),
                    'realtime_reward': self.currentReward,  # / self.m,
                    'wasted_pulls': self.T - sum(self.pulls)}

