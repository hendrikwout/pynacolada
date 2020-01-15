'''

purpose: some example functions

'''

import math
import numpy as np

def moving_average(a, n=3) : 
    cumsum = np.cumsum(a, dtype=float,axis=-1) 
    ret = np.zeros_like(cumsum)*np.nan 
    ret[...,math.ceil(n/2):math.ceil(-n/2)] = cumsum[...,n:] - cumsum[...,:-n] 
 
    ret = ret/n 
    for i in range(0,math.ceil(n/2)): 
            ret[...,i] = (cumsum[...,i+math.floor(n/2)])/(i+math.floor(n/2)+1) 
 
    for i in range(math.ceil(-n/2),0): 
            ret[...,i] = (cumsum[...,-1] - cumsum[...,-1+i-math.floor(n/2)])/(-(i-math.floor(n/2))) 
 
    return ret[:] 
