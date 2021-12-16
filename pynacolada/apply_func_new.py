import numpy as np
import math
import xarray as xr
import os
import netCDF4 as nc4
import pandas as pd
from tqdm import tqdm
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG)
def apply_func(func,xarrays,dims_apply, method_dims_no_apply='outer',filenames_out = None, attributes = None,maximum_input_memory_chars = 10240*1024*100 ,squeeze_apply_dims = False,release=False,output_dims={},transpose_hack=True,tempfile_dir=False,initialize_array=None,copy_coordinates=True):
    logger = logging.getLogger()
    if type(xarrays).__name__ != 'tuple':
        xarrays = [xarrays]

    dims_transposed = {}
    dimslib = {}
    dims_no_apply = []

    xarraydims = []
    xarrayname = []

    logger.debug('Determining dimension names over which the function is applied')
    for ixarray_in,xarray in enumerate(xarrays):
        concat_outer_dim = (type(xarray).__name__ == 'list')
        if concat_outer_dim:
            xarrayname.append(xarray[0].name)
            xarraydims.append(xarray[0].dims)
        else:
            xarrayname.append(xarray.name)
            xarraydims.append(xarray.dims)

        reference_input_xarray = 0
        if ixarray_in == reference_input_xarray:
            dims_apply_def = []
            if type(dims_apply).__name__ in ['list', 'tuple']:
                for dim in dims_apply:
                    if type(dim).__name__ == 'int':
                        logger.debug('Assigning ' + xarraydims[ixarray_in][dim] + ' for dims_apply integer ' + str(dim) +
                              ' according to dimension order of first input array' + xarray[ixarray_in].name)
                        dims_apply_def.append(xarraydims[ixarray_in][dim])
                    else:
                        dims_apply_def.append(dim)
            else:
                raise ValueError('dims_apply needs to be a tuple or list, not a ' + type(dims_apply).__name__ + '.')

    logger.debug('Building default dimensions library describing the entire space with unfolded dimensions over all input arrays '+str(xarray))

    logger.debug('Add first the dimensions over which the operator is being applied. For each dimension name, we keep only track of the longest among the different input arrays.')
    for ixarray_in,xarray in enumerate(xarrays):
        concat_outer_dim = (type(xarray).__name__ == 'list')
        for dim in dims_apply_def:
            if concat_outer_dim:
                if idim == 0:
                    dimslib_temp = xr.concat([xarray_file[dim] for xarray_file in xarray], dim=dim)
                else:
                    dimslib_temp = xarray[0][dim]
                if (dim not in dimslib[dim]) or (len(dimslib_temp) > len(dimslib[dim])):
                    dimslib[dim] = dimlib_temp

    logger.debug('Adding remaining dimensions over which the operator is not being applied. We also check if we have consistent dimensions among the input arrays.')
    for ixarray_in, xarray in enumerate(xarrays):
        concat_outer_dim = (type(xarray).__name__ == 'list')
        for idim,dim in enumerate(xarraydims[ixarray_in]):
            if concat_outer_dim:
                if idim == 0:
                    dimslib_temp = xr.concat([xarray_file[dim] for xarray_file in xarray],dim=dim)
                else:
                    dimslib_temp = xarray[0][dim]

            else:
                dimslib_temp = xarray[dim]

            if (dim not in dimslib.keys() or (len(dimslib[dim]) == 1)):
                dimslib[dim] = dimslib_temp
            else:

                if not ((len(dimslib_temp) == 1) or (dim in dims_apply_def) or (len(dimslib_temp) == len(dimslib[dim])) ):
                    raise ValueError('dimension '+str(dimslib_temp)+' of xarray '+str(xarray) +'is not consistent with dimension '+dimslib[dim]+' occurring in other input arrays. Please check. ')
        logger.debug('dimslib result: '+str(dimslib))

    logger.debug('Determining dimensions of input arrays over which the operator or function is not applied but dupplicated ')
    for ixarray_in, xarray in enumerate(xarrays):
        concat_outer_dim = (type(xarray).__name__ == 'list')
        for ixarray_in, xarray in enumerate(xarrays):
            dims_transposed[xarrayname[ixarray_in]] = []
        for dim in dims_apply_def:
            if dim in xarraydims[ixarray_in]:
                dims_transposed[xarrayname[ixarray_in]] += [dim]

        # adding dimensions over which the function is not applied but duplicated
        # - as outer dimensions (default, should be the fastest way when posisble)
        # - or as inner dimensions (default)
        idim = 0
        for dim in xarraydims[ixarray_in]:
            if dim not in dims_transposed[xarrayname[ixarray_in]]:
                dims_transposed[xarrayname[ixarray_in]].insert(idim,dim)
                idim +=1

                if dim not in dims_no_apply:
                    dims_no_apply += [dim]
                # elif method_dims_no_apply == 'inner':
                #     dims_transposed[xarray[name]].append(dim)

    #import pdb; pdb.set_trace()
    logger.debug('Determining the total space size, chunk sizes, the number of chunks, and the list of chunks')
    total_space_size = np.prod([dimslib[d].shape[0] for d in dims_no_apply+dims_apply_def])
    number_of_chunks = math.ceil(total_space_size*len(xarrays)/maximum_input_memory_chars)

    min_chunk_size = np.prod([dimslib[d].shape[0] for d in dims_apply_def])
    chunk_size = max(min_chunk_size,math.ceil(total_space_size/number_of_chunks))
    chunk_size = min(chunk_size,total_space_size)

    number_of_chunks = total_space_size/chunk_size

    first = True
    chunks =np.arange(0,total_space_size,chunk_size,dtype=int)
    logger.debug('total_space_size/chunks/Setting output filenames')

    logger.debug('Setting output filenames')
    if filenames_out is not None:
        filenames_out_temp = []
        for filename_out in filenames_out:
            if not tempfile_dir:
                filenames_out_temp.append(filename_out)
                logger.debug("Dump output directly to final destination: "+filenames_out_temp[-1])
            else:
                logger.debug("Using temporary output dir, eg., good for working with network file systems")
                if (tempfile_dir is None) or (tempfile_dir is True):
                    filenames_out_temp.append(tempfile.mktemp(suffix='.nc',dir=None))
                    logger.debug("Using temporary output in default tempfile_dir: "+filenames_out_temp[-1])
                else:
                    filenames_out_temp.append(tempfile.mktemp(suffix='.nc',dir=tempfile_dir))
                    logger.debug("Using temporary output in specified tempfile_dir: "+filenames_out_temp[-1])

    logger.debug('Start loop over data chunks')
    for ichunk,chunk_start in tqdm(list(enumerate(chunks))):
        ##print('processing chunk: ', ichunk,'(',chunk_start, ') /',len(chunks), ' (',chunks[-1],')')
        
        arrays_chunk_transposed = []

        chunk_end = min(chunk_start + chunk_size, total_space_size)

        #determin starting end ending point of current chunk in dims_no_apply-space
        this_idxs_start = []
        this_idxs_end = []
        for idim,dim in enumerate(dims_no_apply):
            inner_chunk_size = np.prod([dimslib[d].shape[0] for d in (dims_no_apply[idim+1:]+dims_apply_def)])

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

                for idim,dim in enumerate(dims_apply_def):
                    if dim in xarraydims[ixarray_in]:
                        xarray_chunk_part_select[dim] = range(0,dimslib[dim].shape[0])
                        xarray_chunk_part_dims.append(dim)

                xarray_chunk_part_select_def = {}
                for dim,select in xarray_chunk_part_select.items():

                    # remove explicit selection of full dimension ranges to hopefully speed up xarray selection
                    # we also exclude selection where dimension size of current array is 1
                    if not ((select[0] == 0) and ((select[-1] == (dimslib[dim].shape[0]-1)) or (len(xarray['time']) == 1))):
                        xarray_chunk_part_select_def[dim] = select

                ##print('reading part of chunk of array',str(ipart)+'/'+str(len(idxs_chunk_parts_current)), ichunk,ixarray_in)
                if concat_outer_dim:
                    outer_start = xarray_chunk_part_select[xarray[0].dims[0]][0]
                    outer_end = xarray_chunk_part_select[xarray[0].dims[0]][-1]+1
                    
                    ##print(outer_start,outer_end)
                    ##print('Warning: it is supposed that the input data file list is in sequential order of the outer dimensions.')

                    tempdatalist = []
                    current_outer_pos = 0
                    for ifile,xarray_file in enumerate(xarray):
                        next_outer_pos = current_outer_pos + xarray_file.shape[0]
                        if (current_outer_pos < outer_end) and (next_outer_pos > outer_start):
                            current_file_start = max(outer_start,current_outer_pos) - current_outer_pos
                            current_file_end = min(outer_end,next_outer_pos) - current_outer_pos
                            ##print('reading ifile',ixarray_in,ifile,current_file_start,current_file_end)

                            xarray_chunk_part_select_def[xarray[0].dims[0]] = range(current_file_start,current_file_end)
                            # import pdb;pdb.set_trace()
                            #print(xarray_chunk_part_select_def)
                            tempdatalist.append(np.ascontiguousarray(xarray_file.isel(xarray_chunk_part_select_def).values))
                        current_outer_pos = next_outer_pos

                    array_chunk_part_transposed.append(np.concatenate(tempdatalist,axis=0).transpose([xarraydims[ixarray_in].index(dim) for dim in xarray_chunk_part_dims]))
                    ##print('reading chunk and transposing: ',xarraydims[ixarray_in],'->',xarray_chunk_part_dims)
                    for tempdata in tempdatalist:
                        del tempdata
                    del tempdatalist
                    #import pdb; pdb.set_trace()
                    # print(array_chunk_part_transposed.shape)
                else:
                    try:
                        array_chunk_part_transposed.append(np.ascontiguousarray(xarray.isel(xarray_chunk_part_select_def).values.transpose([xarraydims[ixarray_in].index(dim) for dim in xarray_chunk_part_dims])))
                    except:
                        import pdb; pdb.set_trace()
                    ##print('reading chunk and transposing: ',xarraydims[ixarray_in],'->',xarray_chunk_part_dims)
                shape_out = [1]
                next_idim = 0
                for idim,dim in enumerate(xarray_chunk_part_dims):
                    
                    if dim in dims_no_apply:
                        shape_out[0] *= array_chunk_part_transposed[-1].shape[idim]
                        next_idim = idim+1

                # make modification here to have input xarray chunk in original dimension format!!!
                shape_out += [dimsize for dimsize in array_chunk_part_transposed[-1].shape[next_idim:]]
                array_chunk_part_transposed[-1].shape = shape_out
                ##print('Part of chunk read : ', ixarray_in,ipart)
            ##print('Concatenating chunk for input array',ixarray_in)
            arrays_chunk_transposed.append(np.concatenate(array_chunk_part_transposed,axis=0))
            del array_chunk_part_transposed
            
        ##print('applying function on input chunks')
                
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
            variables_out = []
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
#                        for idim_apply,dim in list(enumerate(dims_apply_def))[::-1]:
                        #dim_found = False
                        # while (not dim_found) and (idim_apply < len(dims_apply_def)):
                        dim = dims_apply_def[idim_apply]
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
                        # while (not dim_found) and (idim_apply < len(dims_apply_def)):
                        #     if (not squeeze_apply_dims) or (shapes_out_transposed[iarray][idim] != 1):
                        if (len(dimslib[dim]) == shapes_out_transposed[iarray][idim]):
                            dims_out_transposed[iarray][idim] = dim
                            if dim in output_dims.keys():
                                coords_out_transposed[iarray][idim] = output_dims[dim]
                            else:
                                coords_out_transposed[iarray][idim] = dimslib[dim]
                        else:
                            dims_out_transposed[iarray][idim] = dim
                            if dim in output_dims.keys():
                                coords_out_transposed[iarray][idim] = output_dims[dim]
                            else:
                                coords_out_transposed[iarray][idim] = range(0,shapes_out_transposed[iarray][idim])
                        dim_found=True
                        idim_apply +=1

                        #     # # end correction
                        #    idim_apply +=1

                        #idim_apply = 0
                        # while (not dim_found) and (idim_apply < len(dims_apply_def)):
                        # # for idim_apply,dim_apply in list(enumerate(dims_apply_def)):
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

            dims_out = []
            for ixarray_out, dim_out_transposed in enumerate(dims_out_transposed):
                # transpose back according to original input

                dims_out.append([None] * len(dims_out_transposed[ixarray_out]))
                coords_out = [None] * len(coords_out_transposed[ixarray_out])
                idims_out = 0
                for ixarray_in, xarray in enumerate(xarrays):
                    for dim in xarraydims[ixarray_in]:
                        if (dim in dims_out_transposed[ixarray_out]) and (dim not in dims_out[ixarray_out]):
                            dims_out[ixarray_out][idims_out] = dim
                            coords_out[idims_out] = coords_out_transposed[ixarray_out][dims_out_transposed[ixarray_out].index(dim)]
                            idims_out += 1
                idims_out_check_point = idims_out

                for idims_out in range(idims_out_check_point, len(dims_out_transposed[ixarray_out])):
                    dims_out[ixarray_out][idims_out] = dims_out_transposed[ixarray_out][idims_out]
                    coords_out[idims_out] = coords_out_transposed[ixarray_out][idims_out]
                #shapes_out_transposed[iarray]

                if filenames_out is not None:
                    xrtemp = xr.Dataset()
                    for idim,dim in enumerate(dims_out[ixarray_out]):
                        xrtemp[dim] = coords_out[idim]
                        # ncouts[iarray].createDimension(dim,shapes_out_transposed[iarray][idim])
                        # ncouts[iarray].createVariable(dim,'d',(dim,),)
                        # ncouts[iarray].variables[dim][:] = coords_out_transposed[iarray][idim]
                    fnout = filenames_out_temp[ixarray_out] #'testing_'+str(iarray)+'.nc'
                    if copy_coordinates:
                        coordinates_template = xarrays[0].coords
                        for coord in coordinates_template: 
                            if coord not in dims_apply_def:
                                xrtemp.coords[coord] = coordinates_template[coord]

                    ##print('writing to file: '+fnout)
                    os.system('mkdir -p '+os.path.dirname(fnout))
                    os.system('rm '+fnout)
                    if initialize_array is not None:
                        xrtemp = initialize_array(xrtemp)
                    # import pdb; pdb.set_trace()
                    xrtemp.to_netcdf(fnout)
                    xrtemp.close()
                    ncouts.append(nc4.Dataset(fnout,'a'))

                    variables_out.append('__xarray_data_variable__')
                    if attributes is not None:
                        for attribute_key,attribute_value in attributes[ixarray_out].items():
                            if attribute_key == 'variable':
                                ##print('setting variable name',variables_out[ixarray_out])
                                variables_out[ixarray_out] = attribute_value

                    ncouts[ixarray_out].createVariable(variables_out[ixarray_out],'f',dims_out[ixarray_out],)

                    # dims_out_def = dims_out_transposed[iarray]
                    # shape_out_def = shapes_out_transposed[iarray]
                    # dims_out_def = [dims_out_def [ :len(dims_no_apply)]+[i for i in dims_out_def[len(dims_no_apply):] if i !=1]
                else:
                    raise ValueError('may have broken. Code needs to be checked.')
                    xarrays_out_transposed.append(
                            xr.DataArray(np.zeros(shapes_out_transposed[iarray])*np.nan,
                                         dims=dims_out[ixarray_out],
                                         coords=coords_out
                            )
                    )
                logger.debug(dims_out)

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
                for dim in dims_apply_def:
                    if (dim in dims_out_transposed[iarray]):# and (not (squeeze_apply_dims and (shapes_out_transposed[iarray][idim]) == 1)):
#                        xarray_chunk_part_select[dim] = range(0,dimslib[dim].shape[0])
                        xarray_chunk_part_select[dim] = range(0,chunk_out_parts_transposed.shape[1+idim])
                        xarray_chunk_part_dims.append(dim)
                        idim += 1

                next_pos_chunk_part_no_apply = pos_chunk_part_no_apply + extent_chunk_part_no_apply
                temp = chunk_out_parts_transposed[pos_chunk_part_no_apply:next_pos_chunk_part_no_apply]
                temp.shape = [ len(dim_select) for dim_select in xarray_chunk_part_select.values()]
                ##print('writing part of chunk of array: ', str(ipart)+'/'+str(len(idxs_chunk_parts_current)),ichunk,iarray)

                #remove explicit selection of full dimension ranges to hopefully speed up xarray selection
                xarray_chunk_part_select_def = {}
                for dim,select in xarray_chunk_part_select.items():
                    if not ((select[0] == 0) and (select[-1] == (dimslib[dim].shape[0]-1))):
                        xarray_chunk_part_select_def[dim] = select
                # import pdb; pdb.set_trace()

                # def mapdims(dims_in,dims_out):
                #     return [dims_out.index(dim) for dim in dims_in] 

                def mapdims(b,a):
                    return [(a.index(b[i]) if b[i] in a else -1) for i in range(len(b))]

                if filenames_out is not None:
                    # logger.debug(list(xarray_chunk_part_select.keys()),dims_out[iarray])
                    # logger.debug(mapdims(dims_out[iarray],list(xarray_chunk_part_select.keys())))

                    ncouts[iarray].variables[variables_out[iarray]][tuple([xarray_chunk_part_select[dim] for dim in dims_out[iarray]])] = np.ascontiguousarray(temp).transpose(mapdims(dims_out[iarray], list(xarray_chunk_part_select.keys())))
                else:
                    xarrays_out_transposed[iarray][xarray_chunk_part_select_def] = np.ascontiguousarray(temp)
                pos_chunk_part_no_apply = next_pos_chunk_part_no_apply
                del temp
            del chunk_out_parts_transposed
        del chunks_out_parts_transposed

    if filenames_out is not None:
        xarrays_out = []
        if attributes is not None:
            for incout,ncout in enumerate(ncouts):
                for attribute_key,attribute_value in attributes[incout].items():
                    if attribute_key is not 'variable':
                        # if attribute_key in ['source_variable','cdf']:
                        #     import pdb; pdb.set_trace()
                        try:
                            if (type(attribute_value) == bool) or (type(attribute_value).__name__ == 'bool_'): 
                                attribute_value_def = np.array(attribute_value,dtype='i1')
                            else:
                                attribute_value_def = attribute_value
                            ncouts[incout][variables_out[incout]].setncattr(attribute_key,attribute_value_def)
                        except:
                            import pdb; pdb.set_trace()

        for incout,ncout in enumerate(ncouts):
            ncouts[incout].close()

            fnout = filenames_out_temp[incout] #'testing_'+str(iarray)+'.nc'
            xarrays_out.append(xr.open_dataset(fnout)[variables_out[incout]])

    if (not release) or (filenames_out is not None):
        for ixarray_out,xarray_out in enumerate(xarrays_out):
             # transpose back according to original input

             # dims_out = [None]*len(dims_out_transposed[ixarray_out])
             # idims_out = 0
             # for ixarray_in,xarray in enumerate(xarrays):
             #     for dim in xarraydims[ixarray_in]:
             #         if (dim in dims_out_transposed[ixarray_out]) and (dim not in dims_out):
             #             dims_out[idims_out] = dim
             #             idims_out += 1
             # idims_out_check_point = idims_out

             # for idims_out in range(idims_out_check_point,len(dims_out)):
             #     dims_out[idims_out] = dims_out_transposed[ixarray_out][idims_out]
             # 
             # def argindex(a,b):
             #    return [(a.index(b[i]) if b[i] in a else -1) for i in range(len(b))]
             # pivot = argindex(dims_out,dims_out_transposed[ixarray_out])
             # if np.any(np.array(pivot[1:]) <  np.array(pivot[:-1])):
             #     print('start transposing of output ',dims_out_transposed[ixarray_out], '->',dims_out)
             #     if (filenames_out is not None):
             #        if transpose_hack:
             #            print('work around. Transposing data on disk according to input data')
             #            xarrays_out_transposed[ixarray_out].close()
             #            fnout = filenames_out_temp[ixarray_out] #'testing_'+str(iarray)+'.nc'
             #            fnout_def =filenames_out[ixarray_out]
             #            os.system('mkdir -p '+os.path.dirname(fnout_def))
             #            os.system('mv '+fnout+' '+fnout+'_temp')
             #            os.system('ncpdq -a '+(','.join(dims_out))+' '+fnout+'_temp '+fnout)
             #            os.system('rm '+fnout_def)
             #            os.system('mv '+ fnout + ' ' +fnout_def)
             #            #xarrays_out_transposed_temp = xr.open_dataset(fnout+'_temp')[variables_out[ixarray_out]].transpose(*dims_out)
             #            #xarrays_out_transposed_temp.to_netcdf(fnout)
             #            #xarrays_out_transposed_temp.close()
             #            os.system('rm '+fnout+'_temp')
             #            if not release:
             #                xarrays_out.append(xr.open_dataset(fnout_def)[variables_out[incout]])
             #        else:
             #        xarrays_out[ixarray_out].close()
             #        fnout = filenames_out_temp[ixarray_out] #'testing_'+str(iarray)+'.nc'
             #        fnout_def =filenames_out[ixarray_out]
             #        os.system('mkdir -p '+os.path.dirname(fnout_def))
             #        os.system('rm '+fnout_def)
             #        os.system('mv '+fnout+' '+fnout_def)#+'_temp')
             #        # os.system('ncpdq -a '+(','.join(dims_out))+' '+fnout+'_temp '+fnout_def)
             #        # os.system('rm '+fnout+'_temp')
             #        if not release:
             #            xarrays_out.append(xr.open_dataset(fnout_def)[variables_out[incout]])
             #            # xarrays_out.append(xarray_out_transposed)
             #            #xarrays_out.append(xr.open_dataset(fnout)[variables_out[incout]])
             #     else:
             #        # xarrays_out_transposed[ixarray_out]
             #        # fnout = filenames_out_temp[ixarray_out] #'testing_'+str(iarray)+'.nc'
             #        # fnout_def =filenames_out[ixarray_out]
             #        # os.system('mkdir -p '+os.path.dirname(fnout_def))
             #        # os.system('mv '+fnout+' '+fnout_def)#+'_temp')

             #        #next line could probably go inside if statement
             #        if not release:
             #            xarrays_out.append(xarray_out_transposed.transpose(*dims_out))

             # else:
             if (filenames_out is not None):
                 xarrays_out[ixarray_out].close()
                 fnout = filenames_out_temp[ixarray_out] #'testing_'+str(iarray)+'.nc'
                 fnout_def =filenames_out[ixarray_out]
                 os.system('mkdir -p '+os.path.dirname(fnout_def))
                 if (fnout != fnout_def):
                    os.system('mv '+fnout+' '+fnout_def)#+'_temp')
                 #next line could probably go inside if statement
                 if not release:
                    xarrays_out[ixarray_out] = xr.open_dataset(fnout_def)[variables_out[incout]]

             # if not release:
             #    xarrays_out.append(xarray_out_transposed)

             # del xarray_out_transposed
        # del xarrays_out_transposed

        if not release:
            if list_output:
                return xarrays_out
            else:
                return xarrays_out[0]


#timeloop =  {{'time':   year} : {'time':year == datatime.astype('datetime64[Y]')} for years in np.unique(datatime.astype('datetime64[Y]'))}

# output will be:

#global_arrayout[0]  = {{'time':   year} : array_out[0]}
def apply_func_per_group(groups,func,xarrays,dims_apply, method_dims_no_apply='outer', maximum_input_memory_chars = 1024*1024*2 ):
    if type(xarrays).__name__ != 'tuple':
        xarrays = (xarrays,)

    global_arrayout = []
    global_dimsout = {}
    first = True
    for dimsout,seldimsin in tqdm(groups):
        #import pdb; pdb.set_trace()

        xarrays_in = []
        for xarray in xarrays:
            seldimsin_this_array = {}
            concat_outer_dim = (type(xarray).__name__ == 'list')

            #xarraydims = []
            if concat_outer_dim:
                # xarrayname.append(xarray[0].name)
                xarraydims = xarray[0].dims
            else:
                # xarrayname.append(xarray.name)
                xarraydims = xarray.dims


            for dimnamein,seldimin in seldimsin.items():

                if dimnamein in xarraydims:
                   if concat_outer_dim and (xarraydims.index(dimnamein) == 0):
                       seldimsin_this_array[dimnamein] = []
                       pos = 0
                       for xarr in xarray:
                           next_pos = pos + len(xarr[dimnamein])
                           seldimsin_this_array[dimnamein].append(seldimin[pos:next_pos])
                           pos = next_pos

                   else:
                       seldimsin_this_array[dimnamein] = seldimin

            if concat_outer_dim:
                xarrays_in.append([])
                for ixarr,xarr in enumerate(xarray):
                    include_this_arr = True
                    seldimsin_this_arr = {}

                    for dimnamein,seldimin in seldimsin_this_array.items():
                        if type(seldimsin_this_array[dimnamein]).__name__ == 'list':
                            seldimsin_this_arr[dimnamein] = seldimsin_this_array[dimnamein][ixarr]                            
                            include_this_arr = np.sum(seldimsin_this_arr[dimnamein]) != 0.
                        else:
                            seldimsin_this_arr[dimnamein] = seldimsin_this_array[dimnamein]
                            

                    if include_this_arr:
                        xarrays_in[-1].append(xarr[seldimsin_this_arr])

                
            else:
                xarrays_in.append(xarray[seldimsin_this_array])

        xarrays_in = tuple(xarrays_in)
        #print(xarrays_in)
        temp = apply_func(func,xarrays_in,dims_apply, maximum_input_memory_chars=maximum_input_memory_chars)

        if type(temp).__name__ not in ['tuple','list']:
            list_output = False
            temp = (temp,)
        else:
            list_output = True

        for itmp,tmp in enumerate(temp):
            if first:
                global_arrayout.append([])
            global_arrayout[itmp].append(tmp)

        if len(dimsout.keys()) > 1:
            logger.debug('number of output dimensions of more than 1 for grouping is not implemented yet.')
            raise
        for dimnameout,dimvalue in dimsout.items():
            if first:
                global_dimsout[dimnameout] = []
            global_dimsout[dimnameout].append(dimvalue)
        first = False

    # for dimnameout,dimvalues in global_dimsout.items():
    #     import pdb;pdb.set_trace()
    #     global_dimsout[dimnameout] = xr.DataArray(global_dimsout[dimnameout],dims=(dimnameout,))

    global_out_def = []
    dimnameout = list(global_dimsout.keys())[0]
    for iglob,glob_array in enumerate(global_arrayout):
        global_out_def.append(xr.concat(glob_array,pd.Index(global_dimsout[dimnameout],name=dimnameout)))

    if list_output:
        return tuple(global_out_def)
    else:
        return global_out_def[0]
