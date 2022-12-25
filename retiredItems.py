import numpy as np
import matplotlib.pyplot as plt
import fixedArms as fA
import time
from copy import deepcopy
import generateArms as gA


def plot_marketSim(K_, T_, m_vals_, rews, stdevs, params_):
    numOpt_, alpha__, bestRew = params_['numOpt'], params_['alpha'], params_['bestReward']
    numArmDist_, totSims_ = params_['numArmDists'], params_['totalSim']
    if numOpt_ == 0:
        numOpt_ = 'no'
    plt.figure(figsize=(7, 5), dpi=100)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    colors = ['red', 'purple', 'mediumseagreen', 'magenta', 'navy', 'blue']
    labels = ['ADA-ETC', 'RADA-ETC', 'ETC', 'NADA-ETC', 'UCB1-s', 'UCB1']

    counter = 0
    for keys in rews.keys():
        plt.plot(m_vals_.astype('str'), rews[keys], color=colors[counter], label=labels[counter])
        plt.errorbar(m_vals_.astype('str'), rews[keys], yerr=stdevs[keys], color=colors[counter],
                     fmt='o', markersize=4, capsize=4)
        counter += 1
    plt.plot(m_vals_.astype('str'), bestRew, color='darkgreen', linestyle='--', label='Best')

    plt.ylabel('Reward', fontsize=13)
    plt.xlabel('m', fontsize=13)

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1.02))
    plt.savefig('res/marketSim_' + str(numOpt_) + 'OptArms_meanK' + str(K_) + '_meanT' + str(T_) + '_alpha' +
                str(alpha__) + '_sim' + str(totSims_) + '_armDist' + str(numArmDist_) + '.eps',
                format='eps', bbox_inches='tight')

    plt.cla()



def sim_small_mid_large_m(armMeansArray_, arrayK_, arrayT_, m_, ucbPart_, pullDiv, alg):
    output = {'reward': -1e8, 'regret': 1e8}
    if m_ == 1:
        if alg == 'ada' or alg == 'rada':
            output = fA.ADAETC(armMeansArray_, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'nada':
            output = fA.NADAETC(armMeansArray_, 1, arrayK_, arrayT_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.UCB1_stopping(armMeansArray_, 1, arrayK_, arrayT_, ucbPart_, verbose=False)
        elif alg == 'ucb1':
            output = fA.naiveUCB1(armMeansArray_, 1, arrayK_, arrayT_, verbose=False)
        elif alg == 'etc':
            output = fA.ETC(armMeansArray_, 1, arrayK_, arrayT_, verbose=False, pullDiv=pullDiv)
    else:
        if alg == 'ada':
            output = fA.m_ADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'rada':
            output = fA.RADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'nada':
            output = fA.m_NADAETC(armMeansArray_, 1, arrayK_, arrayT_, m_, ucbPart_, verbose=False)
        elif alg == 'ucb1s':
            output = fA.m_UCB1_stopping(armMeansArray_, 1, arrayK_, arrayT_, m_, ucbPart_, verbose=False)
        elif alg == 'ucb1':
            output = fA.m_naiveUCB1(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False)
        elif alg == 'etc':
            output = fA.m_ETC(armMeansArray_, 1, arrayK_, arrayT_, m_, verbose=False, pullDiv=pullDiv)
    reward = output['reward']
    regret = output['regret']
    return {'reward': reward, 'regret': regret}


def sample_K_T_streams(numStreams_, totalPeriods_, meanK_, meanT_, m_vals_, geom=False):
    K_list_stream, T_list_stream = {}, {}
    for st in range(numStreams_):
        K_list_, T_list_ = np.zeros(totalPeriods_), np.zeros(totalPeriods_)
        for s in range(totalPeriods_):
            while True:
                if geom:
                    sample_K = np.random.geometric(1 / meanK_, 1)
                else:
                    sample_K = int(np.random.poisson(meanK_, 1))
                if sample_K >= max(2, 2 * max(m_vals_) / totalPeriods_):
                    K_list_[s] = sample_K
                    break
            while True:
                if geom:
                    sample_T = np.random.geometric(1 / meanT_, 1)
                else:
                    sample_T = int(np.random.poisson(meanT_, 1))
                if sample_T > 5 * sample_K:
                    T_list_[s] = sample_T
                    break
        K_list_stream[str(st)], T_list_stream[str(st)] = K_list_, T_list_

    return {'K_list_stream': K_list_stream, 'T_list_stream': T_list_stream}


def marketSim(meanK_, meanT_, numArmDists_, numStreams_, totalPeriods_, m_vals_,
              alpha__, endSim_, algs=None, ucbPart_=2, numOptPerPeriod=0, pullDiv_=1):
    if algs is None:
        algs = {'ada': {}, 'rada': {}, 'nada': {}, 'ucb1s': {}, 'ucb1': {}, 'etc': {}}
    start_ = time.time()
    res = {}  # used to store the reward for each (K, T)-pair stream and algorithm
    resreg = {}  # used to store the regret for each (K, T)-pair stream and algorithm
    stdev = {}  # used to store the standard deviation of rewards for each (K, T)-pair stream and algorithm
    best_rew = {}  # used to store the best reward for each m value and instance

    # a single instance will run with multiple simulations
    # a single instance will have a fixed list of K and T for each period as well as arm means
    # with each instance, ADA-ETC or m-ADA-ETC will be called
    # the ultimate average reward will be calculated based on what these models return and will be averaged over
    # simulations for that instance
    # no need for a global instance generation, the sequence can follow as
    # depending on the number of instances to create, create one instance and following that
    # run however many simulations you need to run and report an average 'average reward' out of this step
    # then the final analysis will be made over averaging these instances

    # fix numStreams_ many K and T streams, pay attention to the least number of arms required per period
    sampleStreams = sample_K_T_streams(numStreams_, totalPeriods_, meanK_, meanT_, m_vals_, geom=False)
    K_list_stream = sampleStreams['K_list_stream']
    T_list_stream = sampleStreams['T_list_stream']
    print("GO")

    # run the simulation for each (K, T)-pair
    # fix (K, T) and following it, fix an instance with the correct number of arms, following the period-based K values
    # here we have the option of (1) either generation one optimal arm per period (by keeping oneOptPerPeriod = True)
    # which generates all the required arms across all periods, selects the best totalPeriods_-many of them and places
    # one of each to every period, and the rest of the arms required for the periods are placed randomly; or
    # (2) generate everything randomly
    for st in range(numStreams_):
        res['stream_' + str(st)] = {}
        resreg['stream_' + str(st)] = {}
        stdev['stream_' + str(st)] = {}
        best_rew['stream_' + str(st)] = {}

        K_list_ = np.array(K_list_stream[str(st)])
        T_list_ = np.array(T_list_stream[str(st)])

        for a in range(numArmDists_):
            res['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            resreg['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            stdev['stream_' + str(st)]['arm_' + str(a)] = deepcopy(algs)
            best_rew['stream_' + str(st)]['arm_' + str(a)] = np.zeros(len(m_vals_))

            if (a + 1) % 5 == 0:
                print("Arm dist number ", str(a + 1), ", K&T stream number", str(st + 1), ", time ",
                      str(time.time() - start_), " from start.")

            # generate arm means, either each period has one optimal arm (across all periods) or no such arrangement
            armInstances_ = gA.generateArms_marketSim(K_list_, T_list_, totalPeriods_, alpha__, numOptPerPeriod)['arms']

            # run multiple simulations for varying values of m
            # if m = 1, every period is run individually and ADA-ETC is called using corresponding (K, T) pair
            # if m > 1, then multiple periods are combined and m-ADA-ETC is called using corresponding (K, T) pair which
            # for this case is summation of different K's and T's.
            for i in range(len(m_vals_)):
                # decide on the number of calls to simulation module and the corresponding m value
                m_val = m_vals_[i]  # say this is 3

                stdev_local = {}
                for keys in algs.keys():
                    res['stream_' + str(st)]['arm_' + str(a)][keys]['m = ' + str(m_val)] = 0
                    resreg['stream_' + str(st)]['arm_' + str(a)][keys]['reg: m = ' + str(m_val)] = 0
                    stdev_local[keys] = []

                calls = int(totalPeriods_ / m_val)  # and this is 6 / 3 = 2, i.e., totalPeriods_ = 6
                for p in range(calls):  # so this will be 0, 1
                    indices = np.zeros(m_val)  # should look like 0, 1, 2;        3,  4,  5
                    for j in range(m_val):
                        indices[j] = p * m_val + j  # so this does the below example
                        # mval = 3; p = 0, j = 0 => 0; p = 0, j = 1 => 1; p = 0, j = 2 => 2
                        # mval = 3; p = 1, j = 0 => 3; p = 1, j = 1 => 4; p = 1, j = 2 => 5

                    # properly combine K and T values if needed
                    K_vals_, T_vals_ = 0, 0
                    for j in range(m_val):
                        K_vals_ += int(K_list_[int(indices[j])])
                        T_vals_ += int(T_list_[int(indices[j])])

                    # depending on the K values, extract the correct set of arms
                    arms_ = np.zeros((1, int(K_vals_)))
                    col_start = 0
                    for j in range(m_val):
                        col_end = int(col_start + K_list_[int(indices[j])])
                        arms_[0, col_start:col_end] = armInstances_[str(int(indices[j]))]
                        col_start += int(K_list_[int(indices[j])])

                    # get the best reward for this m value, and (K, T)-pair
                    best_rew['stream_' + str(st)]['arm_' + str(a)][i] += \
                        np.mean(np.sort(arms_[0])[-m_val:]) * T_vals_ / totalPeriods_

                    # run the multiple simulations on different algorithms
                    for alg_keys in algs.keys():
                        for t in range(endSim_):
                            run = sim_small_mid_large_m(arms_, np.array([K_vals_]), np.array([T_vals_]),
                                                        m_val, ucbPart_, pullDiv_, alg=alg_keys)
                            rew = run['reward'].item()
                            stdev_local[alg_keys].append(rew)
                            rew /= (calls * int(endSim_))
                            reg = run['regret'].item() / (calls * int(endSim_))
                            # .item() is used to get the value, not as a 1-dim array, i.e., fixing the type
                            res['stream_' + str(st)]['arm_' + str(a)][alg_keys]['m = ' + str(m_val)] += rew
                            resreg['stream_' + str(st)]['arm_' + str(a)][alg_keys]['reg: m = ' + str(m_val)] += reg

                    # get the standard deviation on rewards
                    for alg_keys in algs.keys():
                        stdev['stream_' + str(st)]['arm_' + str(a)][alg_keys]['m = ' + str(m_val)] = \
                            np.sqrt(np.var(np.array(stdev_local[alg_keys])) / int(len(stdev_local[alg_keys])))

    # storing results for each algorithm for each m value, across different (K, T)-pair streams (if multiple)
    rewards, stdevs, regrets = deepcopy(algs), deepcopy(algs), deepcopy(algs)
    best_rews_by_m = np.zeros(len(m_vals_))
    normalizer = numArmDists_ * numStreams_
    for j in range(len(m_vals_)):
        m_val = m_vals_[j]
        for keys in algs.keys():
            rewards[keys]['m = ' + str(m_val)] = 0
            regrets[keys]['reg: m = ' + str(m_val)] = 0
            stdevs[keys]['m = ' + str(m_val)] = 0
        for st in range(numStreams_):
            for a in range(numArmDists_):
                best_rews_by_m[j] += best_rew['stream_' + str(st)]['arm_' + str(a)][j] / normalizer
                for kys in algs.keys():
                    rewards[kys]['m = ' + str(m_val)] += \
                        res['stream_' + str(st)]['arm_' + str(a)][kys]['m = ' + str(m_val)] / normalizer
                    regrets[kys]['reg: m = ' + str(m_val)] += \
                        resreg['stream_' + str(st)]['arm_' + str(a)][kys]['reg: m = ' + str(m_val)] / normalizer
                    stdevs[kys]['m = ' + str(m_val)] += \
                        stdev['stream_' + str(st)]['arm_' + str(a)][kys]['m = ' + str(m_val)] / np.sqrt(normalizer)

    # used to pass the rewards and standard deviations to the plotting module as arrays
    alg_rews = {'ada': [], 'rada': [], 'etc': [], 'nada': [], 'ucb1s': [], 'ucb1': []}
    alg_stdevs = {'ada': [], 'rada': [], 'etc': [], 'nada': [], 'ucb1s': [], 'ucb1': []}

    # printing results for each algorithm
    print("numArmDist", numArmDists_, "; alpha", alpha__, "; sims", endSim_,
          "; meanK", meanK_, "; meanT", meanT_, "; numStreams", numStreams_)
    print("How many optimal arm(s) per period?", numOptPerPeriod, "; total periods", totalPeriods_)
    for keys in algs.keys():
        print('--' * 20)
        print(keys + ' results ')
        print('Rewards:')
        print('Best: ', best_rews_by_m)
        for i in range(len(m_vals_)):
            rew = round(rewards[keys]['m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", rew)
            alg_rews[keys].append(rew)
        print()
        print('Standard deviation of rewards')
        for i in range(len(m_vals_)):
            stdev = round(stdevs[keys]['m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", stdev)
            alg_stdevs[keys].append(stdev)
        print()
        print('Regrets')
        for i in range(len(m_vals_)):
            reg = round(regrets[keys]['reg: m = ' + str(m_vals_[i])], 5)
            print("m =", str(m_vals_[i]), ": ", reg)

    print("Done after ", str(round(time.time() - start_, 2)), " seconds from start.")
    params_ = {'numOpt': numOptPerPeriod, 'alpha': alpha__, 'bestReward': best_rews_by_m,
               'numArmDists': numArmDists_, 'totalSim': endSim_}

    plot_marketSim(meanK_, meanT_, m_vals_, alg_rews, alg_stdevs, params_)


# market-like simulation
numArmDists = 50
alpha_ = 0
ucbPart = 2
endSim = 50

meanK = 10  # we will have totalPeriods-many streams, so mean can be set based on that
meanT = 100
numStreams = 1  # number of different K & T streams in total
algorithms = None  # {'etc': {}}  # go for {'rada': {}, 'ucb1': {}} if only want rada-etc and ucb1
totalPeriods = 120
m_vals = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
# np.array([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72])
marketSim(meanK, meanT, numArmDists, numStreams, totalPeriods, m_vals, alpha_, endSim,
          algs=algorithms, ucbPart_=ucbPart, numOptPerPeriod=0, pullDiv_=1)

#
numOpt_a, alphaa__, bestRew = 0, 0, np.array([90.07213815, 92.59938646, 93.39374349, 93.73486892, 93.97043933,
                                            94.0894349, 94.30820747, 94.43713625, 94.49161108, 94.59121827,
                                            94.65095265, 94.68146116, 94.7349182, 94.76393759, 94.82260638,
                                            94.86688588])
# np.array([57.63921225, 58.09828513, 58.25679452, 58.34456167, 58.3925086,
#                    58.41499546, 58.45850016, 58.47262187, 58.49864398, 58.50797159,
#                    58.52839221, 58.53400504, 58.54421696, 58.55064725, 58.56038251,
#                    58.56788171])  # alpha 0.4
numArmDist_, totSims_ = 25, 20
if numOpt_a == 0:
    numOpt_a = 'no'
plt.figure(figsize=(7, 5), dpi=100)
plt.rc('axes', axisbelow=True)
plt.grid()

colors = ['red', 'blueviolet', 'purple', 'mediumseagreen', 'magenta', 'navy', 'blue']
labels = ['ADA-ETC', 'OPT-ETC', 'RADA-ETC', 'ETC', 'NADA-ETC', 'UCB1-s', 'UCB1']

counter = 0
m_vals_a = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
rews = {'ada': {}, 'opt_etc': {}}  # 'rada': {}, 'nada': {}, 'ucb1s': {}, 'ucb1': {}, 'etc': {}, 'opt_etc': {}}
stdevs_ = {'ada': {}, 'opt_etc': {}}  # 'rada': {}, 'nada': {}, 'ucb1s': {}, 'ucb1': {}, 'etc': {}, 'opt_etc': {}}
rews['ada'] = np.array([51.32943, 60.4334, 65.26442, 67.42425, 68.05835, 67.71952, 69.54367, 71.33542, 72.41117,
                        72.14152, 72.19973, 72.35738, 72.73828, 72.18322, 72.59565, 71.76992])
# rews['ada'] = np.array([32.17757, 36.63277, 38.71403, 40.99943, 41.95187, 42.12973, 42.00072, 43.68332, 44.51362,
#                         44.32922, 44.22873, 44.2054, 44.20298, 44.13875, 44.11738, 44.39707])  # alpha 0.4
# rews['rada'] = np.array([])
# rews['nada'] = np.array([])
# rews['ucb1s'] = np.array([])
# rews['ucb1'] = np.array([])
# rews['etc'] = np.array([])
rews['opt_etc'] = np.array([63.20743, 63.40168, 63.39247, 63.27337, 62.86485, 63.22418, 63.0947, 63.09198,
                            63.1359, 63.22332, 62.92507, 62.75525, 63.06253, 62.81763, 63.45027, 62.90392])
# rews['opt_etc'] = np.array([44.6695, 43.70068, 43.53602, 43.3729, 43.39148, 43.44452, 43.2978, 43.42033,
#                             43.46813, 43.41533, 43.2737, 43.42508, 43.4373, 43.38105, 43.28718, 43.52812])  # alpha 0.4

stdevs_['ada'] = np.array([1.32675, 1.24578, 1.25757, 1.35701, 1.374, 1.30531, 1.31018, 1.44739, 1.42141, 1.38197,
                          1.27042, 1.44965, 1.14074, 1.43284, 1.20149, 0.7353])
# stdevs['ada'] = np.array([0.71314, 0.74494, 0.80601, 0.83626, 0.77886, 0.74037, 0.83694, 0.83649, 0.70549, 0.76869,
#                           0.78584, 0.69665, 0.7839, 0.81162, 0.74732, 0.75122])  # alpha 0.4
# stdevs['rada'] = np.array([])
# stdevs['nada'] = np.array([])
# stdevs['ucb1s'] = np.array([])
# stdevs['ucb1'] = np.array([])
# stdevs['etc'] = np.array([])
stdevs_['opt_etc'] = np.array([1.97907, 1.91385, 1.88656, 1.89189, 1.93851, 1.85687, 1.81291, 1.89977, 1.74638,
                              1.8414, 1.62312, 1.7868, 1.91892, 1.54576, 1.53786, 1.43834])
# stdevs['opt_etc'] = np.array([0.83746, 0.86868, 0.80796, 0.86119, 0.84294, 0.82214, 0.78842, 0.80942,
#                               0.77766, 0.80262, 0.75093, 0.77724, 0.76996, 0.67477, 0.65417, 0.62138])  # alpha 0.4

for keys_ in rews.keys():
    plt.plot(m_vals_a.astype('str'), rews[keys_], color=colors[counter], label=labels[counter])
    plt.errorbar(m_vals_a.astype('str'), rews[keys_], yerr=stdevs_[keys_], color=colors[counter],
                 fmt='o', markersize=4, capsize=4)
    counter += 1
plt.plot(m_vals_a.astype('str'), bestRew, color='darkgreen', linestyle='--', label='Best')

plt.ylabel('Reward', fontsize=13)
plt.xlabel('m', fontsize=13)

plt.legend(loc="upper left", bbox_to_anchor=(1, 1.02))
plt.savefig('res/marketSim_' + str(numOpt_a) + 'OptArms_meanK' + str(10) + '_meanT' + str(100) + '_alpha' +
            str(alphaa__) + '_sim' + str(totSims_) + '_armDist' + str(numArmDist_) + '.eps',
            format='eps', bbox_inches='tight')

plt.cla()