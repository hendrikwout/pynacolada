'''

purpose: vectorized functions that can be used with apply_func

'''

import math
import numpy as np
import logging
import xarray as xr



def extend_grid_longitude(longitude,x=None,return_index=False):
    """
    purpose: extend longitude to have a full -180 to 360 domain. This makes selection of locations and domains more easy.

    """

    select_longitude_left = longitude >= 170
    longitude_list = []
    if x is not None:
        x_list = []
        x_list = []

    longitude_index_list = []
    select_longitude_left_index = np.where(select_longitude_left)[0]
    longitude_index_list.append(select_longitude_left_index)

    longitude_left =longitude[select_longitude_left] - 360
    if longitude_left.shape != (0,):
        longitude_list.append(longitude_left)
        if x is not None:
            x_list.append(x[..., select_longitude_left])

    longitude_list.append(longitude)
    if x is not None:
        x_list.append(x)

    longitude_index_list.append(np.array(range(len(longitude))))

    select_longitude_right = longitude < 10.
    longitude_right =longitude[select_longitude_right] + 360
    if longitude_right.shape != (0,):
        longitude_list.append(longitude_right)
        if x is not None:
            x_list.append(x[..., select_longitude_right])

    select_longitude_right_index = np.where(select_longitude_right)[0]
    longitude_index_list.append(select_longitude_right_index)

    longitude_extended = np.concatenate( longitude_list, axis=-1)
    longitude_index_extended = np.concatenate( longitude_list, axis=-1)
    output = [longitude_extended,]

    if x is not None:
        x_extended = np.concatenate(x_list, axis=-1)
        output.append(x_extended)

    if return_index == True:
        longitude_extended_index = np.concatenate(longitude_index_list, axis=-1)
        output.append(longitude_extended_index)

    if len(output) >1:
        return tuple(output)
    else:
        return output[0]


def extend_crop_interpolate(
        x,
        grid_input,
        grid_output,
        interpolation=True,
        return_grid_output=False,
        debug=False,
        border_pixels=5,
        ascending_lat_lon = False,
        tolerance_for_grid_match = 1.e-9
    ):
    """
    purpose:
        perform area cropping and/or interpolation. But also auto-extending the input domain so that
        one can always choose longitude ranges between -180 and 360 degrees.

    input arguments:
        border_pixels: include extra number of pixels at the borders of the domain to ensure consistent interpolation
    """

    grid_input_latitude_spacing = np.abs(np.median(np.ravel(grid_input[0][1:] - grid_input[0][:-1])))
    grid_input_longitude_spacing = np.abs(np.median(np.ravel(grid_input[1][...,1:] - grid_input[1][...,:-1])))

    grid_output_latitude_spacing = np.abs(np.median(np.ravel(grid_output[0][1:] - grid_output[0][:-1])))
    grid_output_longitude_spacing = np.abs(np.median(np.ravel(grid_output[1][...,1:] - grid_output[1][...,:-1])))

    latitude_bottom_input = np.min(grid_output[0]) - grid_input_latitude_spacing*border_pixels #+ grid_output_latitude_spacing/2.
    latitude_top_input = np.max(grid_output[0]) + grid_input_latitude_spacing*border_pixels #- grid_output_latitude_spacing/2.

    grid_input_longitude_extended,grid_input_longitude_extended_index = \
        extend_grid_longitude(grid_input[1],return_index=True)

    longitude_left_input  = np.min(grid_output[1]) - grid_input_longitude_spacing*border_pixels #+ grid_output_longitude_spacing/2.
    longitude_right_input = np.max(grid_output[1]) + grid_input_longitude_spacing*border_pixels #- grid_output_longitude_spacing/2.

    select_longitude_crop_input_index = \
        (grid_input_longitude_extended >= (longitude_left_input - tolerance_for_grid_match)) & \
        (grid_input_longitude_extended <= (longitude_right_input + tolerance_for_grid_match ))

    longitude_crop_input_index = \
        grid_input_longitude_extended_index[ select_longitude_crop_input_index ]

    longitude_crop_input = grid_input_longitude_extended[select_longitude_crop_input_index]

    latitude_crop_input_index = np.where(
        (grid_input[0] >= latitude_bottom_input) &
        (grid_input[0] <= latitude_top_input)
    )[0]
    latitude_crop_input = grid_input[0][latitude_crop_input_index]

    if x is not None:
        if type(x) is xr.DataArray:
        # x_crop = x[...,latitude_crop_input_index,:][...,longitude_crop_input_index]
            x_crop = x.isel(latitude=latitude_crop_input_index, longitude=longitude_crop_input_index).values
        else:
            x_crop = x.take(latitude_crop_input_index,axis=-2).take(longitude_crop_input_index,axis=-1)

    if ascending_lat_lon == True:
        latitude_sort_index = np.argsort(latitude_crop_input)
        latitude_crop_input = latitude_crop_input[latitude_sort_index]
        longitude_sort_index = np.argsort(longitude_crop_input)
        longitude_crop_input = longitude_crop_input[longitude_sort_index]
        if x is not None:
            if type(x) is xr.DataArray:
                x_crop = x_crop.isel(latitude=latitude_sort_index, longitude=longitude_sort_index).values
            else:
                x_crop = x_crop.take(latitude_sort_index,axis=-2).take(longitude_sort_index,axis=-1)

    # ensure that output grid is inside the cropped input grid
    longitude_left_output = np.max([
        np.min(longitude_crop_input),
        np.min(grid_output[1]) - grid_output_longitude_spacing/2.
    ])
    longitude_right_output = np.min([
        np.max(longitude_crop_input),
        np.max(grid_output[1]) + grid_output_longitude_spacing/2.
    ])

    latitude_bottom_output = np.max([
        np.min(latitude_crop_input),
        np.min(grid_output[0]) - grid_output_latitude_spacing/2. #grid_input_latitude_spacing #+ grid_output_latitude_spacing/2.
    ])
    latitude_top_output = np.min([
        np.max(latitude_crop_input),
        np.max(grid_output[0]) + grid_output_latitude_spacing/2. #grid_input_latitude_spacing #- grid_output_latitude_spacing/2
    ])

    grid_output_revised = []
    grid_output_revised.append(
        grid_output[0][(grid_output[0] >= (latitude_bottom_output - tolerance_for_grid_match)) & (grid_output[0] <= (latitude_top_output + tolerance_for_grid_match))]
    )
    grid_output_revised.append(
        grid_output[1][(grid_output[1] >= (longitude_left_output - tolerance_for_grid_match)) & (grid_output[1] <= (longitude_right_output + tolerance_for_grid_match))]
    )

    if debug == True:
        import pdb; pdb.set_trace()

    if (not interpolation) or (\
           (len(grid_output_revised[0]) == len(latitude_crop_input)) and \
           (not np.any(np.abs(grid_output_revised[0] - latitude_crop_input) >
                        (grid_input_latitude_spacing/10.))) and \
           (len(grid_output_revised[1]) == len(longitude_crop_input)) and \
           ( not np.any(np.abs(grid_output_revised[1] - longitude_crop_input) >
                         (grid_input_longitude_spacing / 10.)))
        ):
        if not interpolation:
            logging.info("I'm keeping original grid and spacing, so skipping "
                         "interpolation and returning cropped field directly.")
            grid_output_revised = (latitude_crop_input,longitude_crop_input)
        else:
            logging.info('output grid is identical to cropped input grid. '
       'Skipping interpolation and returning cropped field directly.')
        if x is not None:
            x_interpolated = x_crop
    else:
        logging.warning(
        'Warning. Making a small gridshift to avoid problems in case of coinciding input and output grid locations in the Delaunay triangulation')
        latitude_crop_input_workaround = np.clip(np.float64(latitude_crop_input + 0.000001814),
        -90., 90)
        longitude_crop_input_workaround = np.float64(longitude_crop_input + 0.00001612)
        meshgrid_input_crop = np.meshgrid(
           latitude_crop_input_workaround,
           longitude_crop_input_workaround,
           indexing='ij'
        )

        if x is not None:
            workaround_2_dim = False
            if len(x_crop.shape) == 2:
                workaround_2_dim = True
                x_crop = x_crop[np.newaxis]
            x_interpolated = interpolate_delaunay_linear(
                x_crop,
                meshgrid_input_crop,
                np.meshgrid(*grid_output_revised,indexing='ij'),
                remove_duplicate_points=True,
                dropnans=True,
                add_newaxes=False
            )
            if workaround_2_dim:
                x_interpolated = x_interpolated[0]
        if debug == True:
            import pdb; pdb.set_trace()
    # x_interpolated = pcd.vectorized_functions.interpolate_delaunay_linear(
    #     x_extended,
    #     meshgrid_coarse,
    #     meshgrid_fine,
    #     remove_duplicate_points=True,
    #     dropnans=True,
    #     add_newaxes=False )
    return_value = []
    if x is not None:
        return_value.append(x_interpolated)


    if debug == True:
        import pdb; pdb.set_trace()
    if return_grid_output:
        return_value.append(grid_output_revised)#(latitude_output,longitude_output)
    else:
        if (len(grid_output_revised[0]) != len(grid_output[0])) or \
                (np.max(np.abs(grid_output_revised[0] - grid_output[0])) >= tolerance_for_grid_match) or \
           (len(grid_output_revised[1]) != len(grid_output[1])) or \
                (np.max(np.abs(grid_output_revised[1] - grid_output[1])) >= tolerance_for_grid_match):
            raise ValueError('Predifined output grid is different from actual output grid, '
                             'so you may need that output. Please set return_output_grid to true.')

    if len(return_value) == 0:
        return_value = None
    elif len(return_value) == 1:
        return_value =  return_value[0]
    else:
        return_value = tuple(return_value)
#    import pdb; pdb.set_trace()
    return return_value


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

def calc_quantiles(vals,bins = 50,axis=-1,stable_start_point = True,stable_end_point=True,cdfs=None,profile=None,start=0.,end=0.999):
    # a special way of filtering nans, since we need to conserve the dimension length for vectorized operation


    if profile == 'uniform':
        # cdfs = [ibin / bins for ibin in range(0,bins+1)]
        cdfs = np.linspace(start,end,bins+1)
    elif profile == 'exponential':
        xvals = np.linspace(np.log(1.-end),np.log(1.-start),bins+1)[::-1] 
        cdfs = [ (1.-np.exp(x)) for x in xvals]
    elif profile is not None:
        raise ValueError ('profile not inplemented')
    
    if cdfs is None:
        raise ValueError ('No cdfs could be obtained')

    # cdfwindow = 12*40
    # splitby = 12*5


    sorted_vals = np.sort(vals,axis=axis)
    lengths = np.sum(np.isnan(vals) == False,axis=axis,keepdims  = True,dtype=int)

    pos = [ np.array( (lengths-1) * cdf,dtype=int) for cdf in cdfs]

    
    # pos = []
    # for cdf in cdfs[:-1]:
    #     #pos.append(np.minimum(np.array(lengths * ibin / bins,dtype=int),lengths-1))
    #     pos.append(np.array(lengths * ibin / bins,dtype=int))
    if stable_end_point:
        # ignore the possible very extreme value of the 100th cdf, and replace with the last previous occuring cdf
        pos[-1] = pos[-2]
    if stable_start_point:
        pos[0] = pos[1]

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

def interpolate_delaunay_linear(values,xylist,uvlist,remove_duplicate_points=False ,dropnans=False,fill_value=np.nan,add_newaxes=True):
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
    from scipy.spatial import Delaunay
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
    
    valout = np.einsum('pnj,nj->pn', np.take(valuesstack, vtx,axis=-1), wts)
    valout[:,np.any(wts < 0, axis=1)] = fill_value
    valout.shape = [element for element in values.shape[:-len(xylist[0].shape)]]+[element for element in uvshape ]

    #new axis are added to conform the output to 4 dimenions
    if add_newaxes:
        return np.array(valout[...,np.newaxis,np.newaxis,:,:])
    else:
        return np.array(valout[...,:,:])
    

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
def lookup_nearest(x_fix, y_fix, x_var):
    '''
    lookup nearest y_fix value for which x_fix is closest to the x_var values. The output 'y_var' will have the same dimension as x_var  array x_var array and  of first elements in pair
    
    '''
    x_fix = x_fix.reshape([1]*(len(x_var.shape) )+list(x_fix.shape)) 
    y_fix = y_fix.reshape([1]*(len(x_var.shape) )+list(y_fix.shape))

    # distances = \
    #     x_fix.reshape(list(x_fix.shape[:])+[1])-\
    #     x_var.reshape(list(x_var.shape[:-1])+[1,x_var.shape[-1]]) # 1,1,1001,151
    distances = np.abs(x_fix - np.expand_dims(x_var,axis=-1))

    x_indices_closest = np.expand_dims(np.argmin(np.abs(distances),axis=-1) ,axis=-1)
    y_var_closest = np.take_along_axis(y_fix,x_indices_closest,axis=-1) 
    print(y_var_closest.shape)
    return y_var_closest[...,0]

def interp1d(x_fix, y_fix, x_var,debug=False,random=True):
    '''
    interpolation along axis, and supports parallel vectorized independent iterpolations on multidimensional slices.

    '''
    #print(x_fix.shape,y_fix.shape,x_var.shape)

    # this function assumes that x_fix is monotonically increasing!

    # x_fix = x_fix.reshape([1]*(len(x_var.shape) - len(x_fix.shape))+list(x_fix.shape))
    # y_fix = y_fix.reshape([1]*(len(x_var.shape) - len(x_fix.shape))+list(y_fix.shape))

    #workaround flawed implementation apply_func
    if x_fix.shape[-2] != 1:
        x_fix = x_fix.reshape(list(x_fix.shape[:-1])+[1]+[x_fix.shape[-1]])
    if y_fix.shape[-2] != 1:
        y_fix = y_fix.reshape(list(y_fix.shape[:-1])+[1]+[y_fix.shape[-1]])
    if x_var.shape[-1] != 1:
        x_var = x_var.reshape(list(x_var.shape)+[1])

    # if 1 not in x_fix.shape:
    #     start_shape_weights = 0
    # else:
    #     start_shape_weights = len(x_fix.shape) - x_fix.shape[::-1].index(1) 

    distances = x_fix - x_var

    # import pdb; pdb.set_trace()
    # distances = \
    #         x_fix.reshape(list(x_fix.shape[:])+[1])-\
    #         x_var.reshape(list(x_var.shape[:-1])+[1,x_var.shape[-1]])

    # x_repeat = np.tile(x_var[...,None], (x_fix.shape[-1],))
    # #distances = np.abs(x_repeat - x_fix)
    # distances = np.abs(x_var[...,None] - x_fix[...,None,:] )

    #x_indices_right = np.expand_dims(x_fix.shape[-1] -1 - np.argmax(distances[...,::-1,:] <=0,axis=-2)[...,::-1],axis=-2)
    
    x_indices_right = np.clip(np.expand_dims(np.argmax(distances >0,axis=-1),axis=-1),1,None)
    # for extrapolation on the right hand side
    x_indices_right[distances[...,-1][...,np.newaxis]<=0] = (y_fix.shape[-1] -1)

    distances_closest_right = np.take_along_axis(distances,x_indices_right,axis=-1) 
    x_indices_left = np.clip(x_indices_right-1,None,distances.shape[-1]-1)
    distances_closest_left = np.take_along_axis(distances,x_indices_left,axis=-1) 
    weights = distances_closest_right/(distances_closest_right-distances_closest_left)
    weights[distances_closest_right==distances_closest_left] = 1.

    # weights = weights.reshape(weights.shape[:-1])
    # x_indices_left = x_indices_left.reshape(x_indices_left.shape[:-1])
    # x_indices_right = x_indices_right.reshape(x_indices_right.shape[:-1])


    y_var_right =np.take_along_axis(y_fix,x_indices_right,axis=-1)
    y_var = np.take_along_axis(y_fix,x_indices_left,axis=-1) * weights + np.take_along_axis(y_fix,x_indices_right,axis=-1) * (1.-weights)

    if random ==True:
        y_var_orig = np.array(y_var)
        y_var_max = np.take_along_axis(y_fix,x_indices_left,axis=-1)

        #maxidx = np.take_along_axis(maxidx[None,...],x_indices_left[...,None],axis=-1)
        #maxidx = x_fix.shape[-1] - np.argmax(x_fix[...,::-1] == x_fix[...,::-1,None],axis=-1)[::-1]
        minidx = np.argmax(x_fix[...,None,:] == x_fix[...,None],axis=-1)

        x_indices_min = np.take_along_axis(minidx[...,None,:],x_indices_left[...,None],axis=-1)[...,0]
        y_var_min = np.take_along_axis(y_fix,np.clip(x_indices_min,0,y_fix.shape[-1]-1),axis=-1)

        weight_rand = np.random.rand(*y_var_min.shape)
        select_for_random_y = ( (x_indices_min) != (x_indices_left)) & (weights == 1.)
        if debug == True:
            import pdb; pdb.set_trace()
        y_var[select_for_random_y] = (y_var_min *weight_rand + y_var_max*(1-weight_rand))[select_for_random_y]
    
    return y_var[...,0]

def interp1d_orig(x_fix, y_fix, x_var):
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



