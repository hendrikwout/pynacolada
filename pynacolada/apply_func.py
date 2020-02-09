import numpy as np
import math
import xarray as xr
import os
import netCDF4 as nc4

def apply_func(func,xarrays,dims_apply, method_dims_no_apply='outer',filenames_out = None, maximum_input_memory_chars = 1024*1024*1024 ,squeeze_apply_dims = False,release=False):
    if type(xarrays).__name__ != 'tuple':
        xarrays = [xarrays]

    dims_transposed = {}
    dimslib = {}
    dims_no_apply = []

    xarraydims = []
    xarrayname = []

    for ixarray_in,xarray in enumerate(xarrays):
        concat_outer_dim = (type(xarray).__name__ == 'list')
        if concat_outer_dim:
            xarrayname.append(xarray[0].name)
            xarraydims.append(xarray[0].dims)
        else:
            xarrayname.append(xarray.name)
            xarraydims.append(xarray.dims)

        for idim,dim in enumerate(xarraydims[ixarray_in]):
            if dim not in dimslib.keys():
                if concat_outer_dim:
                    if idim == 0:
                        dimslib[dim] = xr.concat([xarray_file[dim] for xarray_file in xarray],dim=dim)
                    else:
                        dimslib[dim] = xarray[0][dim]

                else:
                    dimslib[dim] = xarray[dim]
        dims_transposed[xarrayname[-1]] = []
        for dim in dims_apply:

            if dim in xarraydims[-1]:
                dims_transposed[xarrayname[-1]] += [dim]

        # adding dimensions over which the function is not applied but duplicated
        # - as outer dimensions (default, should be the fastest way when posisble)
        # - or as inner dimensions (default)
        idim = 0
        for dim in xarraydims[-1]:
            if dim not in dims_transposed[xarrayname[-1]]:
                if method_dims_no_apply == 'outer':
                    dims_transposed[xarrayname[-1]].insert(idim,dim)
                    idim +=1
                    
                    if dim not in dims_no_apply:
                        dims_no_apply += [dim]


                # elif method_dims_no_apply == 'inner':
                #     dims_transposed[xarray[name]].append(dim)
                else:
                    raise




    total_array_size = np.prod([dimslib[d].shape[0] for d in dims_no_apply+dims_apply])
    number_of_chunks = math.ceil(total_array_size*len(xarrays)/maximum_input_memory_chars)

    min_chunk_size = np.prod([dimslib[d].shape[0] for d in dims_apply])
    chunk_size = max(min_chunk_size,math.ceil(total_array_size/number_of_chunks))
    chunk_size = min(chunk_size,total_array_size)

    number_of_chunks = total_array_size/chunk_size

    first = True
    chunks =np.arange(0,total_array_size,chunk_size,dtype=int)
    for ichunk,chunk_start in enumerate(chunks):
        print('processing chunk: ', ichunk,'(',chunk_start, ') /',len(chunks), ' (',chunks[-1],')')
        
        arrays_chunk_transposed = []

        chunk_end = min(chunk_start + chunk_size, total_array_size)

        #determin starting end ending point of current chunk in dims_no_apply-space
        this_idxs_start = []
        this_idxs_end = []
        for idim,dim in enumerate(dims_no_apply):
            inner_chunk_size = np.prod([dimslib[d].shape[0] for d in dims_no_apply[idim+1:]+dims_apply])

            this_idx_start = math.floor(chunk_start/inner_chunk_size)
            this_idxs_start.append(this_idx_start)
            chunk_start = chunk_start - this_idx_start*inner_chunk_size

            this_idx_end = math.floor(chunk_end/inner_chunk_size)
            this_idxs_end.append(this_idx_end)
            chunk_end = chunk_end - this_idx_end*inner_chunk_size


        shape_no_apply = [dimslib[d].shape[0] for d in dims_no_apply]

        idxs_chunk_parts_current = []
        idxs_chunk_parts_end = []

        # if these indices are equal then this means: select 1 element.
        idxs_chunk_part_current = this_idxs_start.copy()
        idxs_chunk_part_end = this_idxs_start.copy()
        # import pdb;pdb.set_trace()
        for iidx,idx_chunk_part_current in list(enumerate(idxs_chunk_part_current))[::-1]:
            #idx_chunk_part_end = idxs_chunk_part_end[iidx]

            last_moving_index = True
            for iiidx in range(len(idxs_chunk_part_current[:iidx])):
                    if (this_idxs_end[iiidx] - idxs_chunk_part_current[iiidx]) != 0:
                        last_moving_index = False


            if (last_moving_index == False) and (idxs_chunk_part_current[iidx] != 0) and (idxs_chunk_part_current[iidx] < shape_no_apply[iidx]):
                idxs_chunk_part_end[iidx] = shape_no_apply[iidx]-1

                idxs_chunk_parts_current.append(idxs_chunk_part_current.copy())
                idxs_chunk_parts_end.append(idxs_chunk_part_end.copy())


                # for next cycle
                idxs_chunk_part_end[-1] +=1
                for iidx,idx_chunk_part_end in list(enumerate(idxs_chunk_part_end))[::-1]:
                    if idxs_chunk_part_end[iidx]==shape_no_apply[iidx]:
                        if iidx > 0:
                            idxs_chunk_part_end[iidx] = 0
                            idxs_chunk_part_end[iidx-1] += 1
                idxs_chunk_part_current = idxs_chunk_part_end.copy()
        
        #import pdb; pdb.set_trace()
        for iidx,idx_chunk_part_current in list(enumerate(idxs_chunk_part_current)):
            if idx_chunk_part_current < (this_idxs_end[iidx]):
                idxs_chunk_part_end[iidx] = this_idxs_end[iidx]

                if iidx < len(idxs_chunk_part_current):
                    idxs_chunk_part_end[iidx] -= 1
                    for iiidx in range(len(idxs_chunk_part_current))[iidx+1:]:
                        idxs_chunk_part_end[iiidx] = shape_no_apply[iiidx]-1

                idxs_chunk_parts_current.append(idxs_chunk_part_current.copy())
                idxs_chunk_parts_end.append(idxs_chunk_part_end.copy())
                # for next cycle

                idxs_chunk_part_end[-1] +=1
                for iidx,idx_chunk_part_end in list(enumerate(idxs_chunk_part_end))[::-1]:
                    if idxs_chunk_part_end[iidx]==shape_no_apply[iidx]:
                        if iidx > 0:
                            idxs_chunk_part_end[iidx] = 0
                            idxs_chunk_part_end[iidx-1] += 1
                idxs_chunk_part_current = idxs_chunk_part_end.copy()
                
        for ixarray_in,xarray in enumerate(xarrays):
            concat_outer_dim = type(xarray).__name__ == 'list'
            array_chunk_part_transposed = []


            # if dims_no_apply dimensions are not in the array, then we don't
            # need to loop over the specified parts. Otherwise we will have too
            # big input arrays for the function with artificially duplicated
            # data. So we only need to loop over the parts where changing
            # no_apply dimensions are in the current array. So we reasign
            # idxs_chunk_parts values for the dims_no_apply dimensions that are
            # not in the array to a dummy value (zero). Afterwards, we take
            # only the parts with unique indices, so that duplicates are
            # avoided. This is is hard to explain, damnit.
            idxs_chunk_parts_current_this_array = []
            idxs_chunk_parts_end_this_array = []
            for ipart,idxs_chunk_part_current in enumerate(idxs_chunk_parts_current):
                idxs_chunk_parts_current_this_array.append([])
                idxs_chunk_parts_end_this_array.append([])
                for idim,dim in enumerate(dims_no_apply):
                    if dim in xarraydims[ixarray_in]:
                        idxs_chunk_parts_current_this_array[ipart].append(idxs_chunk_parts_current[ipart][idim])
                        idxs_chunk_parts_end_this_array[ipart].append(idxs_chunk_parts_end[ipart][idim])
                    else:
                        idxs_chunk_parts_current_this_array[ipart].append(0)
                        idxs_chunk_parts_end_this_array[ipart].append(0)

            idxs_chunk_parts_current_this_array,idxs_chunk_parts_end_this_array = \
                    np.unique(np.concatenate([np.array(idxs_chunk_parts_current_this_array)[...,np.newaxis],np.array(idxs_chunk_parts_end_this_array)[...,np.newaxis]],axis=-1),axis=0).transpose(2,0,1).tolist()

            for ipart,idxs_chunk_part_current_this_array in enumerate(idxs_chunk_parts_current_this_array):
                idxs_chunk_part_end_this_array = idxs_chunk_parts_end_this_array[ipart]
                xarray_chunk_part_select = {}#[None]*len(xarraydims[ixarray_in])
                xarray_chunk_part_dims = []
                for idim,dim in enumerate(dims_no_apply):
                    idx_chunk_part_current = idxs_chunk_part_current_this_array[idim]
                    idx_chunk_part_end = idxs_chunk_part_end_this_array[idim]
                    if dim in xarraydims[ixarray_in]:
                        xarray_chunk_part_select[dim] = range(idx_chunk_part_current,idx_chunk_part_end+1)
                        xarray_chunk_part_dims.append(dim)

                for idim,dim in enumerate(dims_apply):
                    if dim in xarraydims[ixarray_in]:
                        xarray_chunk_part_select[dim] = range(0,dimslib[dim].shape[0])
                        xarray_chunk_part_dims.append(dim)

                # print(idxs_chunk_parts_current_this_array)
                # print(idxs_chunk_parts_end_this_array)
                # print(xarray_chunk_part_select)
                # print(xarray_chunk_part_dims)
        
                #remove explicit selection of full dimension ranges to hopefully speed up xarray selection
                xarray_chunk_part_select_def = {}
                for dim,select in xarray_chunk_part_select.items():
            
                    if not ((select[0] == 0) and (select[-1] == (dimslib[dim].shape[0]-1))):
                        xarray_chunk_part_select_def[dim] = select


                print('reading part of chunk of array',str(ipart)+'/'+str(len(idxs_chunk_parts_current)), ichunk,ixarray_in)
                if concat_outer_dim:
                    outer_start = xarray_chunk_part_select[xarray[0].dims[0]][0]
                    outer_end = xarray_chunk_part_select[xarray[0].dims[0]][-1]+1
                    
                    print(outer_start,outer_end)
                    print('Warning: it is supposed that the input data file list is in sequential order of the outer dimensions.')

                    tempdatalist = []
                    current_outer_pos = 0
                    for ifile,xarray_file in enumerate(xarray):
                        next_outer_pos = current_outer_pos + xarray_file.shape[0]
                        if (current_outer_pos < outer_end) and (next_outer_pos > outer_start):
                            current_file_start = max(outer_start,current_outer_pos) - current_outer_pos
                            current_file_end = min(outer_end,next_outer_pos) - current_outer_pos
                            print('reading ifile',ixarray_in,ifile,current_file_start,current_file_end)

                            xarray_chunk_part_select_def[xarray[0].dims[0]] = range(current_file_start,current_file_end)
                            # import pdb;pdb.set_trace()
                            #print(xarray_chunk_part_select_def)
                            tempdatalist.append(np.ascontiguousarray(xarray_file.isel(xarray_chunk_part_select_def).values))
                        current_outer_pos = next_outer_pos

                    array_chunk_part_transposed.append(np.concatenate(tempdatalist,axis=0).transpose([xarraydims[ixarray_in].index(dim) for dim in xarray_chunk_part_dims]))
                    print('reading chunk and transposing: ',xarraydims[ixarray_in],'->',xarray_chunk_part_dims)
                    for tempdata in tempdatalist:
                        del tempdata
                    del tempdatalist
                    #import pdb; pdb.set_trace()
                    # print(array_chunk_part_transposed.shape)
                else:
                    array_chunk_part_transposed.append(np.ascontiguousarray(xarray.isel(xarray_chunk_part_select_def).values.transpose([xarraydims[ixarray_in].index(dim) for dim in xarray_chunk_part_dims])))
                    print('reading chunk and transposing: ',xarraydims[ixarray_in],'->',xarray_chunk_part_dims)
                shape_out = [1]
                next_idim = 0
                for idim,dim in enumerate(xarray_chunk_part_dims):
                    
                    if dim in dims_no_apply:
                        shape_out[0] *= array_chunk_part_transposed[-1].shape[idim]
                        next_idim = idim+1
                        
                shape_out += [dimsize for dimsize in array_chunk_part_transposed[-1].shape[next_idim:]]
                array_chunk_part_transposed[-1].shape = shape_out
                print('Part of chunk read : ', ixarray_in,ipart)
            print('Concatenating chunk for input array',ixarray_in)
            arrays_chunk_transposed.append(np.concatenate(array_chunk_part_transposed,axis=0))
            del array_chunk_part_transposed
            
        print('applying function on input chunks')
                
        chunks_out_parts_transposed = func(*arrays_chunk_transposed) #.squeeze_apply_dims()
#        import pdb; pdb.set_trace()

        list_output = True
        if type(chunks_out_parts_transposed).__name__ not in ['tuple','list']:
                list_output = False
                chunks_out_parts_transposed = [chunks_out_parts_transposed]
        shape_out_func = [None]*len(chunks_out_parts_transposed)

        # we need a second shape that indicates the squeeze_apply_dimsd shape. But we need to keep the first one too for detecting the output dimensions of the apply_dims.
        for iarray,chunk_out_parts_transposed in enumerate(chunks_out_parts_transposed):
            if squeeze_apply_dims:
                shape_out_func[iarray] = [chunks_out_parts_transposed[iarray].shape[0]]+[i for i in chunks_out_parts_transposed[iarray].shape[1:] if i !=1]
            else:
                shape_out_func[iarray] = chunks_out_parts_transposed[iarray].shape


        del arrays_chunk_transposed
        if first:
            xarrays_out_transposed = []
            ncouts = []
            dims_out_transposed = []
            shapes_out_transposed = []
            coords_out_transposed = []
            for iarray,chunk_out_parts_transposed in enumerate(chunks_out_parts_transposed):
                shapes_out_transposed.append(shape_no_apply+[dimsize for dimsize in chunk_out_parts_transposed.shape[1:]])
                dims_out_transposed.append([None]*len(shapes_out_transposed[iarray]))
                coords_out_transposed.append([None]*len(shapes_out_transposed[iarray]))

                idim_apply = 0
                for idim,dimsize in enumerate(shapes_out_transposed[iarray]):
                    if idim < len(shape_no_apply):
                        dim = dims_no_apply[idim]
                        dims_out_transposed[iarray][idim] = dim
                        coords_out_transposed[iarray][idim] = dimslib[dim]
                    else:
                        # we inherit the dimension namings from the apply_dims
#                        for idim_apply,dim in list(enumerate(dims_apply))[::-1]:
                        #dim_found = False
                        # while (not dim_found) and (idim_apply < len(dims_apply)):
                        dim = dims_apply[idim_apply]
                        # if (dim not in dims_out_transposed[iarray]) and (not (squeeze_apply_dims and (shapes_out_transposed[iarray][idim]) == 1)) :
                            # if (dim_apply == dim) and (dimslib[dim_apply] != shapes_out_transposed[iarray][idim]) and (shapes_out_transposed[iarray][idim] != 1):
                            #     raise IOError('I was expecting the same dimension length for the output ('+str(+ shapes_out_transposed[iarray][idim]+') as for the input ('+str(+ dimslib[dim_apply]+')  for '+dim+'. Please check.')


                            # # correction!!!!!
                            # dims_out_transposed[iarray][idim] = dim
                            # if (len(dimslib[dim]) == shapes_out_transposed[iarray][idim]):
                            #     coords_out_transposed[iarray][idim] = dimslib[dim]
                            #     # coords_out_transposed[iarray][idim].name = dim
                            # else:
                            #     coords_out_transposed[iarray][idim] = range(0,shapes_out_transposed[iarray][idim])
                        # dim_found = False
                        # while (not dim_found) and (idim_apply < len(dims_apply)):
                        #     if (not squeeze_apply_dims) or (shapes_out_transposed[iarray][idim] != 1):
                        if (len(dimslib[dim]) == shapes_out_transposed[iarray][idim]):
                            dims_out_transposed[iarray][idim] = dim
                            coords_out_transposed[iarray][idim] = dimslib[dim]
                        else:
                            dims_out_transposed[iarray][idim] = dim
                            coords_out_transposed[iarray][idim] = range(0,shapes_out_transposed[iarray][idim])
                        dim_found=True
                        idim_apply +=1

                        #     # # end correction
                        #    idim_apply +=1

                        #idim_apply = 0
                        # while (not dim_found) and (idim_apply < len(dims_apply)):
                        # # for idim_apply,dim_apply in list(enumerate(dims_apply)):
                        #     if (dim not in dims_out_transposed[iarray]) and ((not (squeeze_apply_dims and (shapes_out_transposed[iarray][idim]) == 1)) or (len(dimslib[dim]) == shapes_out_transposed[iarray][idim])):
                        #         dims_out_transposed[iarray][idim] = dim
                        #         coords_out_transposed[iarray][idim] = range(0,shapes_out_transposed[iarray][idim])
                        #         dim_found = True
                        #     idim_apply +=1

                        # if (not dim_found) and (not (squeeze_apply_dims and (shapes_out_transposed[iarray][idim]) == 1)):
                        #     # implementation needed here to make custom dims
                        #     dims_out_transposed[iarray][idim] = 'test'
                        #     coords_out_transposed[iarray][idim] = range(shapes_out_transposed[iarray][idim])
                        #     #raise
                if squeeze_apply_dims:
                    dims_out_transposed_old = list(dims_out_transposed[iarray])
                    shapes_out_transposed_old = list(shapes_out_transposed[iarray])
                    coords_out_transposed_old = list(coords_out_transposed[iarray])
                    dims_out_transposed[iarray] = []
                    shapes_out_transposed[iarray] = []
                    coords_out_transposed[iarray] = []
                    for idim,dim in enumerate(dims_out_transposed_old):
                        if (shapes_out_transposed_old[idim] > 1) or (idim < len(dims_no_apply)):
                            dims_out_transposed[iarray].append(dim)
                            coords_out_transposed[iarray].append(coords_out_transposed_old[idim])
                            shapes_out_transposed[iarray].append(shapes_out_transposed_old[idim])
                
                #shapes_out_transposed[iarray]
                if filenames_out is not None:
                    xrtemp = xr.Dataset()
                    for idim,dim in enumerate(dims_out_transposed[iarray]):
                        xrtemp[dim] = coords_out_transposed[iarray][idim]
                        #import pdb;pdb.set_trace()
                        # ncouts[iarray].createDimension(dim,shapes_out_transposed[iarray][idim])
                        # ncouts[iarray].createVariable(dim,'d',(dim,),)
                        # ncouts[iarray].variables[dim][:] = coords_out_transposed[iarray][idim]

                    fnout = filenames_out[iarray] #'testing_'+str(iarray)+'.nc'
                    print('writing to file: '+fnout)
                    os.system('rm '+fnout)
                    xrtemp.to_netcdf(fnout)
                    xrtemp.close()
                    ncouts.append(nc4.Dataset(fnout,'a'))
                    
                    ncouts[iarray].createVariable('__xarray_data_variable__','f',dims_out_transposed[iarray],)

                    # dims_out_def = dims_out_transposed[iarray]
                    # shape_out_def = shapes_out_transposed[iarray]
                    # dims_out_def = [dims_out_def [ :len(dims_no_apply)]+[i for i in dims_out_def[len(dims_no_apply):] if i !=1]
                else:
                    xarrays_out_transposed.append(
                            xr.DataArray(np.zeros(shapes_out_transposed[iarray])*np.nan,
                                         dims=dims_out_transposed[iarray],
                                         coords=coords_out_transposed[iarray]
                            )
                    )
            print(coords_out_transposed)
            print(dims_out_transposed)
            first = False
        
        for iarray,chunk_out_parts_transposed in enumerate(chunks_out_parts_transposed):
            chunk_out_parts_transposed.shape = shape_out_func[iarray]
            pos_chunk_part_no_apply = 0
            for ipart,idxs_chunk_part_current in enumerate(idxs_chunk_parts_current):
                idxs_chunk_part_end = idxs_chunk_parts_end[ipart]

                xarray_chunk_part_select = {}#[None]*len(xarraydims[ixarray_in])
                xarray_chunk_part_dims = []
                extent_chunk_part_no_apply = 1
                idim = 0
                for dim in dims_no_apply:
                    idx_chunk_part_current = idxs_chunk_part_current[idim]
                    idx_chunk_part_end = idxs_chunk_part_end[idim]
                    if dim in dims_out_transposed[iarray]:
                        xarray_chunk_part_select[dim] = range(idx_chunk_part_current,idx_chunk_part_end+1)
                        xarray_chunk_part_dims.append(dim)
                        extent_chunk_part_no_apply *= idx_chunk_part_end + 1 - idx_chunk_part_current
                        idim += 1

                idim = 0
                for dim in dims_apply:
                    if (dim in dims_out_transposed[iarray]):# and (not (squeeze_apply_dims and (shapes_out_transposed[iarray][idim]) == 1)):
#                        xarray_chunk_part_select[dim] = range(0,dimslib[dim].shape[0])
                        xarray_chunk_part_select[dim] = range(0,chunk_out_parts_transposed.shape[1+idim])
                        xarray_chunk_part_dims.append(dim)
                        idim += 1

                next_pos_chunk_part_no_apply = pos_chunk_part_no_apply + extent_chunk_part_no_apply
                temp = chunk_out_parts_transposed[pos_chunk_part_no_apply:next_pos_chunk_part_no_apply]
                temp.shape = [ len(dim_select) for dim_select in xarray_chunk_part_select.values()]
                print('writing part of chunk of array: ', str(ipart)+'/'+str(len(idxs_chunk_parts_current)),ichunk,iarray)

                #remove explicit selection of full dimension ranges to hopefully speed up xarray selection
                xarray_chunk_part_select_def = {}
                for dim,select in xarray_chunk_part_select.items():
                    if not ((select[0] == 0) and (select[-1] == (dimslib[dim].shape[0]-1))):
                        xarray_chunk_part_select_def[dim] = select
                # import pdb; pdb.set_trace()
                if filenames_out is not None:
                    ncouts[iarray].variables['__xarray_data_variable__'][tuple(xarray_chunk_part_select.values())] = np.ascontiguousarray(temp)
                else:
                    xarrays_out_transposed[iarray][xarray_chunk_part_select_def] = np.ascontiguousarray(temp)
                pos_chunk_part_no_apply = next_pos_chunk_part_no_apply
                del temp
            del chunk_out_parts_transposed
        del chunks_out_parts_transposed


    if filenames_out is not None:
        xarrays_out_transposed = []
        for incout,ncout in enumerate(ncouts):
            ncout.close()
            fnout = filenames_out[incout] #'testing_'+str(iarray)+'.nc'
            if not release:
                xarrays_out_transposed.append(xr.open_dataset(fnout)['__xarray_data_variable__'])

    if not release:
        xarrays_out = [] 
        for ixarray_out,xarray_out_transposed in enumerate(xarrays_out_transposed):
             # transpose back according to original input

             #import pdb; pdb.set_trace()
             dims_out = [None]*len(dims_out_transposed[ixarray_out])
             idims_out = 0
             for ixarray_in,xarray in enumerate(xarrays):
                 for dim in xarraydims[ixarray_in]:
                     if (dim in dims_out_transposed[ixarray_out]) and (dim not in dims_out):
                         dims_out[idims_out] = dim
                         idims_out += 1
             idims_out_check_point = idims_out

             for idims_out in range(idims_out_check_point,len(dims_out)):
                 dims_out[idims_out] = dims_out_transposed[ixarray_out][idims_out]
             
             def argindex(a,b):
                return [(a.index(b[i]) if b[i] in a else -1) for i in range(len(b))]
             pivot = argindex(dims_out,dims_out_transposed[ixarray_out])
             if np.any(np.array(pivot[1:]) <  np.array(pivot[:-1])):
                 print('start transposing of output ',dims_out_transposed[ixarray_out], '->',dims_out)
                 xarrays_out.append(xarray_out_transposed.transpose(*dims_out))
             else:
                 xarrays_out.append(xarray_out_transposed)

             del xarray_out_transposed
        del xarrays_out_transposed

        if list_output:
            return xarrays_out
        else:
            return xarrays_out[0]


#timeloop =  {{'time':   year} : {'time':year == datatime.astype('datetime64[Y]')} for years in np.unique(datatime.astype('datetime64[Y]'))}

# output will be:

#global_arrayout[0]  = {{'time':   year} : array_out[0]}
