# -*- coding: utf-8 -*-
"""
Simple Monte-Carlo search on a continuous domain bounded within [lb,ub]**n
"""
import numpy as np
import matplotlib.pyplot as plt
import objFunctions as fct

def simpleMonteCarlo(n, lb, ub, evals, func=lambda x: x.dot(x)) :
    history = []
    xmin = np.random.uniform(size=n)*(ub - lb) + lb
    fmin = func(xmin)
    history.append(fmin)
    for _ in range(evals) :
        x = np.random.uniform(size=n)*(ub - lb) + lb
        f_x = func(x)
        if f_x < fmin :
            xmin = x
            fmin = f_x
        history.append(fmin)
    return xmin,fmin,history
#
if __name__ == "__main__" :
    lb,ub = -10,10
    n=30
    evals=10**5
    xmin,fmin,history = simpleMonteCarlo(n,lb,ub,evals,fct.WildZumba)
    plt.semilogy(history)
