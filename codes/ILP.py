from pulp import *
import numpy as np




def ILP(P):
    emo_n = len(P)
    m = LpProblem(name='ILP', sense=LpMinimize)

    Y = [LpVariable(f'y{i}', cat=LpBinary) for i in range(emo_n)]
    C = [(-np.log2(P[i]) + np.log2(1 - P[i])) for i in range(emo_n)]

    ## 目标函数
    z = 0
    for i in range(emo_n):
        z += Y[i] * C[i]

    ## 解决目标函数有绝对值问题
    sum_var = LpVariable('sum_var')
    abs_sum_var = LpVariable('abs_sum_var')

    m += abs_sum_var
    m += sum_var == z
    m += abs_sum_var >= sum_var
    m += abs_sum_var >= -sum_var
    m += (lpDot([Y[i] for i in range(emo_n)], [1] * emo_n) >= 1)
    m += (lpDot([Y[i] for i in range(emo_n)], [1] * emo_n) <= 3)

    # 求解
    m.solve()

    res = [int(pulp.value(y)) for y in Y]
    return res

if __name__ == '__main__':
    # P = [0.0884692370891571,0.019895583391189575,0.07140126824378967,0.22996234893798828,0.02351740002632141,0.007670044898986816,
    #      0.39467552304267883,0.00037735700607299805,0.0001621246337890625,0.3239821791648865,0.01828068494796753,0.01294735074043274]

    # P = [0.6, 0.08, 0.42]
    P = [0.071144700050354,0.0031821727752685547,0.003788501024246216,0.0017750561237335205,0.058736056089401245]

    print(ILP(P))