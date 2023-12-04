from amazon_marketSim import *
from mDependentFncs import *


if __name__ == '__main__':

    print('Figures 1a-b and 2a-b: 2-armed bandit with T-dependent gaps')

    mEqOne(K_list_=np.array([2]), T_list_=np.arange(1, 11) * 4000, numArmDists_=50, endSim_=50, alpha__=0,
           numOpt_=1, delt_=0, plots=True, ucbSim=True, improved=False, fixed='no', justUCB='yes',
           NADA=True, NADAucb=2, fn=1)  # 1/T^(2/5)
    mEqOne(K_list_=np.array([2]), T_list_=np.arange(1, 11) * 4000, numArmDists_=50, endSim_=50, alpha__=0,
           numOpt_=1, delt_=0, plots=True, ucbSim=True, improved=False, fixed='no', justUCB='yes',
           NADA=True, NADAucb=2, fn=2)  # 1/T^(1/2)
    #
    #

    #
    print('Figures 3a -- 3d: m=1')
    numArmDists = 200
    endSim = 50
    K_vals = np.array([4, 8])
    T_list = np.arange(1, 11) * 100
    alpha_vals = np.array([0, 0.4])

    for i in range(len(K_vals)):
        K_list = np.array([K_vals[i]])
        for j in range(len(alpha_vals)):
            alpha_ = alpha_vals[j]

            print(" K ", K_list, " alpha", alpha_)
            mEqOne(K_list, T_list, numArmDists, endSim, alpha_,
                   numOpt_=1, delt_=0, plots=True, ucbSim=False, improved=False, fixed='no', justUCB='no', NADA=True)
    #
    #

    #
    print('Figures 4a -- 4c: T=100, 1000 simulations')

    print('Figure 4a: K=2, 1 optimal arm')
    mEqOne_varyGapPlots(K_list_=np.array([2]), T_list_=np.array([100]), endSim_=1, numOpt_=1,
                        generateIns_=1000, rng=101, ucbSim=True, justUCB='no', NADA=True)

    print('Figure 4b: K=4, 1 optimal arm')
    mEqOne_varyGapPlots(K_list_=np.array([4]), T_list_=np.array([100]), endSim_=1, numOpt_=1,
                        generateIns_=1000, rng=101, ucbSim=True, justUCB='no', NADA=True)

    print('Figure 4c: K=4, 2 optimal arms')
    mEqOne_varyGapPlots(K_list_=np.array([4]), T_list_=np.array([100]), endSim_=1, numOpt_=2,
                        generateIns_=1000, rng=101, ucbSim=True, justUCB='no', NADA=True)
    #
    #

    #
    print('Figures 5a -- 5f: General m')
    alpha_vals = np.array([0, 0.4])
    m_vals = np.array([2, 2, 4])
    K_vals = np.array([4, 8, 8])
    T_list = np.arange(1, 11) * 100
    numArmDists = 200
    endSim = 50
    for i in range(len(m_vals)):
        K_list = np.array([K_vals[i]])
        m = m_vals[i]
        for j in range(2):
            alpha_ = alpha_vals[j]

            print("m ", m, " K ", K_list, " alpha", alpha_)
            mGeneral(K_list, T_list, numArmDists, endSim, m, alpha_, numOpt_=1, delt_=0, improved=False,
                     UCBin=True, NADA=True)
    #
    #

    #
    print('Figures 6a -- 6d: Amazon experiment')
    amazon()
    #
    #

    #
    print("Figure 7: m=4, K=8, alpha=0; longer T")
    numArmDists = 200
    endSim = 50
    mGeneral(np.array([8]), np.arange(1, 16) * 200, numArmDists, endSim, 4, 0, numOpt_=1, delt_=0, improved=False,
             UCBin=True, NADA=True)
    #
    #

    #
    print('Figure 8: Amazon experiment with UCB1 in')
    amazon(True)
    #
    #

    #
    print('Figures 9a-b and 10a-b: The market simulation')
    market()
    #
    #
