import numpy as np
# import pandas as pd
import time
import fixedArms as fA
import generateArms as gA

if __name__ == '__main__':
    K_list = np.array([2])
    varyingK = True if len(K_list) > 1 else False
    T_list = np.array([200, 400, 600, 800, 1000])
    numArmDists = 50
    alpha = 0
    startSim = 0
    endSim = 500
    pw = 1 / 2

    # armInstances = gA.generateArms(K_list, T_list, numArmDists, alpha)
    armInstances = gA.generateMultipleArms(K_list, T_list, numArmDists, pw)

    start = time.time()
    fA.naiveUCB1(armInstances, startSim, endSim, K_list, T_list)
    # fA.ETC(armInstances, startSim, endSim, K_list, T_list)
    fA.ADAETC(armInstances, startSim, endSim, K_list, T_list)
    # fA.NADAETC(armInstances, startSim, endSim, K_list, T_list)
    # fA.UCB1_stopping(armInstances, startSim, endSim, K_list, T_list)
    print("took " + str(time.time() - start) + " seconds; arms within " + str(pw) + " power of respective T values")
    print("Larger the ratio of difference between cumulative rewards to the total")
    print("cumulative rewards, better the performance of max objective")
