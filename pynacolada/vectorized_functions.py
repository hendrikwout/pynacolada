'''

purpose: vectorized functions that can be used with apply_func

'''

import math
import numpy as np
from scipy.spatial import Delaunay

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

def calc_quantiles(vals,bins = 50,axis=-1,cdfs=None,stable_end_point = True ,profile=None,start=0.,end=1.):
    # a special way of filtering nans, since we need to conserve the dimension length for vectorized operation
    sorted_vals = np.sort(vals,axis=axis)
    lengths = np.sum(np.isnan(vals) == False,axis=axis,keepdims  = True,dtype=int)

    if profile == 'uniform':
        cdfs = [ibin / bins for ibin in range(0,bins+1)]
    elif profile == 'exponential':
        xvals = np.linspace(np.log(1.-end),np.log(1.-start),bins+1)[::-1] 
        cdfs = [ (1.-np.exp(x)) for x in xvals]
    elif profile is not None:
        raise ValueError ('profile not inplemented')
    
    if cdfs is None:
        raise ValueError ('No cdfs could be obtained')

    pos = [ np.array(lengths * cdf,dtype=int) for cdf in cdfs]
    
    # pos = []
    # for cdf in cdfs[:-1]:
    #     #pos.append(np.minimum(np.array(lengths * ibin / bins,dtype=int),lengths-1))
    #     pos.append(np.array(lengths * ibin / bins,dtype=int))
    if stable_end_point:
        # ignore the possible very extreme value of the 100th cdf, and replace with the last previous occuring cdf
        pos[-1] = pos[-2]

    pos = np.concatenate(pos,axis=axis)
    quantiles = np.take_along_axis(sorted_vals,pos,axis = axis) 
#     print(cdfs)
    
    return (quantiles,quantiles*0+np.array(cdfs).reshape([1]*(len(quantiles.shape)-1)+[np.array(cdfs).shape[-1]])) 

def biascorrect_quantiles(series_biased,series_reference,**kwargs):
    # ,profile='exponential',bins=50,end=0.9999,
    cdfs,quantiles_biased    = calc_quantiles(series_biased,**kwargs)
    cdfs,quantiles_reference = calc_quantiles(series_reference,cdfs=cdfs)
    series_cdf_biased    = interp1d(quantiles_biased,cdfs,series_bias)
    series_cdf_reference = interp1d(quantiles_reference,cdfs,series_reference)

    series_biased_recalc         = interp1d(cdfs,quantiles_biased,cdf_series_biased)
    series_corrected_preliminary = interp1d(cdfs,quantiles_reference,cdf_series_biased)
    series_corrected = series_biased + series_biased_recalc - series_corrected_preliminary

    return series_corrected

def interpolate_delaunay_linear(values,xylist,uvlist,remove_duplicate_points=False ,dropnans=False,fill_value=np.nan):
    d = len(xylist)

    uvshape = uvlist[0].shape
    xystack = np.stack([xy.ravel() for xy in xylist],axis=-1) 
    valuesstack = values.copy()
    valuesstack.shape =  [element for element in values.shape[:-len(xylist[0].shape)]] + [element for element in xystack.shape[0:1]]
    if dropnans:
        nans =[x for x in np.where(~np.isnan(valuesstack[0]))[0]]
        valuesstack = np.take(valuesstack,nans,axis=1)
        xystack = np.take(xystack,nans,axis=0)
    
    # if remove_duplicate_points:
    #     index_input_unique = np.unique(xystack,return_index=True,axis=0)[1]
    #     xystack = np.take(xystack,index_input_unique,axis=0)
    #     valuesstack = np.take(valuesstack,index_input_unique,axis=1)
    uvstack = np.stack([uv.ravel() for uv in uvlist],axis=-1) 

    for i,xy in enumerate(xylist): 
        if xy.shape != xylist[0].shape:
            raise ValueError('Dimension of xylist['+str(i)+'] should be be equal to xylist[0].')

    print( values.shape[:len(xylist[0].shape)] ,  xylist[0].shape)
    if values.shape[-len(xylist[0].shape):] != xylist[0].shape:
        raise ValueError('Inner dimensions of "values" '+str(values.shape)+' should be equal to the dimensions of the arrays in xylist '+ str(xylist[0].shape)+'.')


    uvstackshape = tuple(uvstack.shape)
    axis0shape = 1
    for facdim in uvstack.shape[:-1]:
        axis0shape *= facdim
    uvstack.shape = (axis0shape,uvstack.shape[-1]) 
    # print(uvstack.shape)
    # print(xystack.shape) 
    tri = Delaunay(xystack) 
    # print(uvstack.shape)  
    simplex = tri.find_simplex(uvstack) 
    vertices = np.take(tri.simplices, simplex, axis=0) 
    temp = np.take(tri.transform, simplex, axis=0)                                                                                       
    delta = uvstack - temp[:, d] 
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)                                                                                
    vtx, wts = vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    outeraxisshaperavel = 1
    for facdim in valuesstack.shape[:-1]:
        outeraxisshaperavel *= facdim
    valuesstack.shape = [outeraxisshaperavel] + [valuesstack.shape[-1]] 
    
    # import pdb; pdb.set_trace()
    valout = np.einsum('pnj,nj->pn', np.take(valuesstack, vtx,axis=-1), wts)
    valout[:,np.any(wts < 0, axis=1)] = fill_value
    valout.shape = [element for element in values.shape[:-len(xylist[0].shape)]]+[element for element in uvshape ]
    # import pdb;pdb.set_trace()

    #new axis are added to conform the output to 4 dimenions
    return np.array(valout[...,np.newaxis,np.newaxis,:,:])
    

# def calc_cdf_series(vals,quantiles):
#     cdf_disc = np.zeros(list(vals.shape)+[quantiles.shape[-1]],dtype=bool)
#     for iquantile in range(0,quantiles.shape[-1]):
#         if iquantile < (quantiles.shape[-1]-1):
#             select = ((vals >= quantiles[...,iquantile:iquantile+1]) & (vals < quantiles[...,(iquantile+1):(iquantile+2)])) | (vals == quantiles[...,iquantile:iquantile+1]) 
#         elif iquantile == (quantiles.shape[-1] - 1):
#             select = (vals >= quantiles[...,iquantile:iquantile+1]) 
#         cdf_disc[...,iquantile] = select
#     
#         #ranges = np.concatenate([cdf_disc.argmax(axis=-1)[...,np.newaxis],cdf_disc[...,::-1].argmax(axis=-1)[...,np.newaxis]],axis=-1)
#     
#     # def argmax_reverse(data,*args,**kwargs):
#     #     return data.shape[-1] - data[...::-1].argmax(axis=-1)
#     
#     
#     out = np.round(np.random.uniform(cdf_disc.argmax(axis=-1),cdf_disc.shape[-1]  - (cdf_disc[...,::-1].argmax(axis=-1)+1)))
#     out[np.isnan(vals)] = np.nan
#     
#     return out



def interp1d(x_fix, y_fix, x_var):
    '''
    interpolation along axis, and supports parallel vectorized independent iterpolations on multidimensional slices.

    '''
    print(x_fix.shape,y_fix.shape,x_var.shape)

    # this function assumes that x_fix is monotonically increasing!

    # x_fix = x_fix.reshape([1]*(len(x_var.shape) - len(x_fix.shape))+list(x_fix.shape))
    # y_fix = y_fix.reshape([1]*(len(x_var.shape) - len(x_fix.shape))+list(y_fix.shape))

    # if 1 not in x_fix.shape:
    #     start_shape_weights = 0
    # else:
    #     start_shape_weights = len(x_fix.shape) - x_fix.shape[::-1].index(1) 

    distances = \
            x_fix.reshape(list(x_fix.shape[:])+[1])-\
            x_var.reshape(list(x_var.shape[:-1])+[1,x_var.shape[-1]])

    # x_repeat = np.tile(x_var[...,None], (x_fix.shape[-1],))
    # #distances = np.abs(x_repeat - x_fix)
    # distances = np.abs(x_var[...,None] - x_fix[...,None,:] )

    #x_indices_right = np.expand_dims(x_fix.shape[-1] -1 - np.argmax(distances[...,::-1,:] <=0,axis=-2)[...,::-1],axis=-2)
    x_indices_right = np.expand_dims(np.argmax(distances >0,axis=-2),axis=-2)
    distances_closest_right = np.take_along_axis(distances,x_indices_right,axis=-2) 
    x_indices_left = np.clip(x_indices_right-1,None,distances.shape[-2]-1)
    distances_closest_left = np.take_along_axis(distances,x_indices_left,axis=-2) 
    weights = distances_closest_right/(distances_closest_right-distances_closest_left)
    weights[distances_closest_right==distances_closest_left] = 1.

    weights = weights.reshape(list(weights.shape[:-2])+[weights.shape[-1]])
    x_indices_left = x_indices_left.reshape(list(x_indices_left.shape[:-2])+[x_indices_left.shape[-1]])
    x_indices_right = x_indices_right.reshape(list(x_indices_right.shape[:-2])+[x_indices_right.shape[-1]])

    y_var_right =np.take_along_axis(y_fix,x_indices_right,axis=-1) 
    y_var = np.take_along_axis(y_fix,x_indices_left,axis=-1) * weights + np.take_along_axis(y_fix,x_indices_right,axis=-1) * (1.-weights)
    #import pdb; pdb.set_trace()
    y_var_orig = np.array(y_var)
    y_var_max = np.take_along_axis(y_fix,x_indices_left,axis=-1)

    #maxidx = np.take_along_axis(maxidx[None,...],x_indices_left[...,None],axis=-1)
    #maxidx = x_fix.shape[-1] - np.argmax(x_fix[...,::-1] == x_fix[...,::-1,None],axis=-1)[::-1] 
    minidx = np.argmax(x_fix[...,None,:] == x_fix[...,None],axis=-1) 
    x_indices_min = np.take_along_axis(minidx[...,None,:],x_indices_left[...,None],axis=-1)[...,0]
    y_var_min = np.take_along_axis(y_fix,np.clip(x_indices_min,0,y_fix.shape[-1]-1),axis=-1)

    weight_rand = np.random.rand(*y_var_min.shape)
    select_for_random_y = ( (x_indices_left-1) != (x_indices_min))
    y_var[select_for_random_y] = (y_var_min *weight_rand + y_var_max*(1-weight_rand))[select_for_random_y]
    y_var[np.isnan(x_var)] = np.nan
    return y_var

    # x_indices_left = np.argmax(distances >=0,axis=-2)

    # x_indices_pre = np.expand_dims(np.argmin(np.abs(distances),axis=-2),axis=-2)
    # distances_closest_pre = np.take_along_axis(distances,x_indices_pre,axis=-2)
    # x_indices_2_pre = x_indices_pre-np.array(+np.sign(distances_closest_pre)-(distances_closest_pre==0),dtype=int)
    # x_indices_2 = np.clip(x_indices_2_pre , 0,distances.shape[-2]-1)
    # x_indices = x_indices_pre + x_indices_2 - x_indices_2_pre

    # # we set nan values to zero... we filter them afterwards.
    # invalid_indices = x_indices < 0
    # x_indices[invalid_indices] = 0.
    # x_indices_2[invalid_indices] = 1.

    # distances_closest = np.take_along_axis(distances,x_indices,axis=-2) 
    # distances_closest_2 = np.take_along_axis(distances,x_indices_2,axis=-2) 

    # distances_closest[invalid_indices] = np.nan
    # distances_closest_2[invalid_indices] = np.nan

    # sel_left_2 = distances_closest_2<0
    # distances_closest_left = np.array(distances_closest)
    # distances_closest_left[sel_left_2] = distances_closest_2[sel_left_2]
    # distances_closest_right = np.array(distances_closest)
    # distances_closest_right[~sel_left_2] = distances_closest_2[~sel_left_2]

    # x_indices_left = np.array(x_indices)
    # x_indices_left[sel_left_2] = x_indices_2[sel_left_2]
    # x_indices_right = np.array(x_indices)
    # x_indices_right[~sel_left_2] = x_indices_2[~sel_left_2]


    # minidx = np.argmax(x_fix == x_fix[...,None],axis=-1)
    # 
    # selecting = x_fix == x_fix[:,None]
    # np.take_along_axis(selecting,x_indices_left[:,None],axis=0)


    # minmaxidx = np.concatenate([minidx[:,None],maxidx[:,None]],axis=-1)
    # np.take_along_axis(minidx[None,...],x_indices_left[...,None],axis=-1)
    # np.take_along_axis(minidx[None,...],x_indices_left[...,None],axis=-1)

    # y_indices = np.concatenate([np.take_along_axis(minidx[None,...],x_indices_left[...,None],axis=-1),np.take_along_axis(maxidx[None,...],x_indices_left[...,None],axis=-1)],axis=-1)


    # idx = np.arange(len(x_indices))
    # weights[idx,x_indices] = distances[idx,x_indices-1]
    # weights[idx,x_indices-1] = distances[idx,x_indices]
    # weights /= np.sum(weights, axis=1)[:,None]

    # #y_var = np.dot(weights, y_fix.T)
    # indices_pre = 'abcdefg'[:start_shape_weights]
    # indices_post = 'lmnopqr'[:len(x_fix.shape[start_shape_weights+1:])]
    # np.einsum(indices_pre+'i'+indices_post+',k'+indices_post+' -> '+indices_pre+'ik'+indices_post, weights[...,0,:], y_fix)



