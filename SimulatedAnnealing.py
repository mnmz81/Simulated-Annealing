# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def simpleMonteCarlo(n) :
    """this is hill Climber variation"""
    return 2*np.random.randint(2, size=(n)) - 1

def basicHillClimber(x,n) :
    """this is hill Climber variation"""
    index=np.random.randint(n);
    x[index]=x[index]*-1
    return x

def myVariation(x,n):
    """this is my variation for next"""
    k=0
    temp=[]
    while k<len(x):
        t=[]
        t=x.copy()
        t[k]=t[k]*-1
        temp.append(t)
        k=k+1
    return temp[np.random.randint(n)]

def SwedishPump(b) :
    """this is the function thet we lock for max"""
    """ The correlation function, assumes a numpy vector {-1,+1} as input """
    n = np.size(b)
    E = []
    for k in range(1,n) :
        E.append((b[:n-k].dot(b[k:]))**2)
    return (n**2)/(2*sum(E))


def SimulatedAnnealing(n,max_evals, variation=simpleMonteCarlo, func=SwedishPump, seed=None) :
    T_init=10.0
    T_min=1e-4
    alpha=0.99
    max_internal_runs = 50
    local_state = np.random.RandomState(seed)
    history = []
    xbest = xbase = xmax = 2*np.random.randint(2, size=(n)) - 1
    fbest = fmax = func(xmax)
    eval_cntr = 1
    T = T_init
    history.append(fmax)
    while ((T > T_min) and eval_cntr < max_evals) :
        for _ in range(max_internal_runs):
            x=variation(xmax,n)
            f_x = func(x)
            eval_cntr += 1
            dE =fmax-f_x
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-dE/T) :
                xmax = x
                fmax = f_x
            if dE < 0 :
                fbest=f_x
                xbest=x
            history.append(fmax)
            if np.mod(eval_cntr,int(max_evals/10))==0 :
                print(eval_cntr," evals: fmax=",fmax)
        T *= alpha
    return xbest,fbest,history
#
if __name__ == "__main__" :
    n=100
    evals=10**5
    Nruns=10
    fbest = []
    xbest = []
    for i in range(Nruns) :
        xmax,fmax,history = SimulatedAnnealing(n,evals,myVariation,SwedishPump,i+17)
        plt.semilogy(history)
        plt.show()
        print(i,": max of Swedish Pump found is ", fmax," at location ", xmax)
        fbest.append(fmax)
        xbest.append(xmax)
    print("====\n Best ever: ",max(fbest),"x*=",xbest[fbest.index(max(fbest))])