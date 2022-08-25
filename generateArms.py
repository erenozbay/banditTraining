import numpy as np


def generateArms(K_list, T_list, numArmDists, alpha):
    ncol = sum(K_list) * len(T_list)
    armInstances = np.zeros((numArmDists, ncol))

    for i in range(numArmDists):
        armInstances[i, :] = np.random.uniform(alpha, 1 - alpha, ncol)
    print(armInstances)
    return armInstances


# def generateTwoArms(T_list, numArmDists, pw=1 / 3):
#     ncol = 2 * len(T_list)
#     armInstances = np.zeros((numArmDists, ncol))
#
#     for i in range(numArmDists):
#         arms = np.zeros(ncol)
#         for j in range(len(T_list)):
#             upper = min(0.25, 1 / np.power(T_list[j], pw))
#             diff = np.random.uniform(0, upper, 1) + upper
#             first = 0.5 - diff
#             second = 0.5 + diff
#             arms[j * 2] = first
#             arms[j * 2 + 1] = second
#         armInstances[i, :] = arms
#
#     print(armInstances)
#     return armInstances


def generateMultipleArms(K_list, T_list, numArmDists, pw=1 / 3):
    ncol = sum(K_list) * len(T_list)
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

    print(armInstances)
    return armInstances
