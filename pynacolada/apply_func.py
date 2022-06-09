#import dask.array as da
from argparse import Namespace
import tempfile
import os
import netCDF4 as nc4
from itertools import product
import numpy as np
#import rasterio
import xarray as xr
#from dask.diagnostics import ProgressBar
#from rasterio.windows import Window
import pandas as pd
import numpy as np
import tqdm
import logging
import sys
sys.path.insert(0, 'lib/pynacolada/')
import pynacolada as pcd
#ProgressBar().register()
# if __name__ == '__main__':
#input_file = r'Y:\Unit_RMA\_OneBackup\RMA_RDM\GIS_data\Vlaanderen\BBK_Verharding\2018_beta\BBK2018v01\BBK1_18\BBK1_18.tif'

# input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
# ds = xr.open_dataset(
#     input_file,
#     chunks={
#         'lat':int(ds.shape[0] / 40),
#         'lon':int(ds.shape[1] / 40)}
# )['Band1']

def name_from_pattern(pattern, attributes):
    return ''.join(
        np.array(
            list(
                zip(pattern.split('"')[::2],
                    [attributes[key] for key in
                     pattern.split('"')[1::2]
                     ] + [''])
            )
        ).ravel()
    )


def get_coordinates_attributes(coords):
    coordinates_attributes = {}
    if 'time' in coords.keys():
        # if ('time' not in dict_index.keys()) or (dict_index['time'] is None) or (
        #         type(dict_index['time']).__name__ == 'float') and (np.isnan(dict_index['time']).any()):
        #     print('Guessing time coordinate from DataArray')
        #     # is month type

        # monthly spacing
        if coords['time'] is None:
            coordinates_attributes['time'] = None
        elif str(coords['time'].dtype).startswith('int') :
            coordinates_attributes['time'] = 'integer_'+str(coords['time'].values[0])+'_'+str(coords['time'].values[-1])
        elif np.apply_along_axis(lambda y: np.sum((y[1:] - y[:-1] != 1), 0), 0,
                               np.vectorize(lambda x: int(x[:4]) * 12 + int(x[5:7]))(
                                   coords['time'].values.astype('str'))).item() == 0:
            coordinates_attributes['time'] = \
                'monthly_' + str(coords['time'][0].values)[:7] + '_' + str(coords['time'][-1].values)[:7]
        # also monthly
        elif (not np.any(~(np.vectorize(lambda x: x[8:])(
                coords['time'].values.astype('str')) == '01T00:00:00.000000000'))):
            coordinates_attributes['time'] = \
                'monthly_' + str(coords['time'][0].values)[:7] + '_' + str(coords['time'][-1].values)[:7]
        elif not np.any((coords['time'][2:-1].values - coords['time'][1:-2].values) != np.array(86400000000000,
                                                                                                dtype='timedelta64[ns]')):
            # daily
            coordinates_attributes['time'] = 'daily_' + np.datetime_as_string(coords['time'][0].values,
                                                                              unit='D') + '_' + np.datetime_as_string(
                coords['time'][-1].values, unit='D')
        elif not np.any((coords['time'][2:-1].values - coords['time'][1:-2].values) != dt.timedelta(days=1)):
            coordinates_attributes['time'] = \
                'daily_' + str(coords['time'][0].values)[:10] + '_' + str(coords['time'][-1].values)[:10]
        else:
            coordinates_attributes['time'] = 'irregular'
            logging('warning. No time dimension found')

        # DataArray.attrs['time'] = dict_index['time']

    # # renamings = {'lat': 'latitude', 'lon': 'longitude'}
    # for key, value in renamings.items():
    #     if key in coords.keys():
    #         coor = DataArray.rename({key: value})

    # # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
    # space_coordinates = list(coords.keys())
    # for key in self.lib_dataarrays.index.names:
    #     if key in space_coordinates:
    #         space_coordinates.remove(key)
    space_coordinates = ['latitude','longitude']

    spacing = {}
    for dim,coord in coords.items():
        if dim in space_coordinates:
            spacing_temp = (coords[dim].values[1] - coord[dim].values[0])
            if not np.any(
                    coords[dim][1:].values != (coords[dim].values[:-1] + spacing_temp)):
                spacing[dim] = str(coords[dim][0].values) + ',' + str(
                    coords[dim][-1].values) + ',' + str(spacing_temp)
            else:
                spacing[dim] = 'irregular'
    dict_index_space = [key + ':' + str(value) for key, value in spacing.items()]
    dict_index_space = '_'.join(dict_index_space)
    coordinates_attributes['space'] = dict_index_space

    return coordinates_attributes


def identical_xarrays(xarray1,xarray2):
    return (xarray1.shape == xarray2.shape) and (not (np.any(xarray1 != xarray2)))

def sort_dict_by_keys(dict_input, dict_input_sort_keys):
    """
    Sort a dictionary so that the keys takes a predifined order. The order doesn't need to have all the keys included.

    :param dict_input: input dictionary to be sorted
    :param dict_input_sort_keys: order of keys. Not all keys of dict_input need to be given. The remaining ones will just keep their position.
    :return: sorted version of dict_input according to dict_input_sort_keys
    """
    dim_indices = []
    for dimname in dict_input.keys():
        if dimname in dict_input_sort_keys:
            dim_indices.append(dict_input_sort_keys.index(dimname))
        else:
            dim_indices.append(None)
    logging.debug('created index order list for ' + str(dict_input.keys()) + ' to ' + str(
        dict_input_sort_keys) + ': ' + str(dim_indices))

    iidx_pause = 0
    for ia, dimname_idx in enumerate(dim_indices):
        if dimname_idx is None:
            dim_indices[ia] = ia
            iidx_pause = ia + 1
        else:
            dim_indices[ia] = iidx_pause + dim_indices[ia]

    logging.debug('... and with the Nones filled: ' + str(dim_indices))

    # for dimname in dict_input_sort_keys:
    #     dim_indices.append(list(dict_input.keys()).index(dimname))

    # import pdb; pdb.set_trace()
    # # for dim_index in list(range(len(dim_indices)+1, len(dict_input)+1)):
    # #     dim_indices.append(dim_index)
    # for dim_index in range(len(dict_input)):
    #     if dim_index not in dim_indices:
    #         dim_indices.append(dim_index)


    # import pdb; pdb.set_trace()
    dict_output = {
        #list(dict_input.keys())[index]: dict_input[list(dict_input.keys())[index]] for index in
        index:dict_input[index] for index in [x for _,x in sorted(zip(dim_indices,dict_input.keys()))] }

    return dict_output

def apply_func(
        func,
        xarrays,
        dims_apply_names = [],
        xarrays_output_filenames = [],
        maximum_memory_size_bytes = 2 * 10 ** 7 ,
        output_dimensions={},
        xarrays_output_dimensions = [],
        tempfile_dir=False,
        ignore_memory_limit = False,
        overwrite_output_filenames = False,
        pass_missing_output_coordinates = False,
    ):
    #input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'

    # xarrays_output_dimensions = [ #input
    #     {
    #         'time': {'coords': ds2.time},
    #         'latitude': {'coords': ds.latitude, 'chunksize': 4500},
    #         'longitude': {'coords': ds.longitude, 'chunksize': 4500},
    #     }
    # ]

    logging.info('collecting the dimensions occuring in the xarrays over which the function is not applied')
    dims_no_apply  = {}

    for ixarray,xarray in enumerate(xarrays):
        for dimname in xarray.dims:
            if (dimname not in dims_no_apply) and (dimname not in dims_apply_names):
                dims_no_apply[dimname] = xarray[dimname]
            if (dimname in dims_no_apply) and (not identical_xarrays(xarray[dimname],dims_no_apply[dimname])):
                raise ValueError('dimension '+dimname+' of xarray number '+str(ixarray)+' is not the same as previously detected dimensions.')


    # if output_dimensions is None:
    #     output_dimensions_sort_keys = list(output_dimensions.keys())
    # else:
    #     output_dimensions_sort_keys = []

    logging.debug('adding missing output dimensions from specified xarrays_coordinates_output')
    if xarrays_output_dimensions != []:
        output_dimensions_new = {}
        for dimdict in xarrays_output_dimensions:
            for dimname in dimdict.keys():
                try:
                 if dimname not in output_dimensions_new.keys():
                     logging.info('adding missing output_dimensions from xarrays_output_dimensions for '+dimname+': '+dimdict[dimname])
                     output_dimensions_new[dimname] = dimdict[dimname]
                except:
                    import pdb; pdb.set_trace()
        logging.debug('overriding values and order of previous output_dimensions.')

        if output_dimensions is None:
            output_dimensions = {}

        for output_dimensions_orig in output_dimensions.keys():
            output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
        output_dimensions = sort_dict_by_keys(output_dimensions_new,list(output_dimensions.keys()))


    output_dimensions_new = {}
    for dimname,coordinates in dims_no_apply.items():

        if dimname not in output_dimensions_new.keys():
            logging.info('adding no apply dimensions to the default output_dimensions '+dimname)
            output_dimensions_new[dimname] = {'coords':coordinates}

    if output_dimensions is None:
        output_dimensions = {}

    for output_dimensions_orig in output_dimensions.keys():
        output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
    output_dimensions = sort_dict_by_keys(output_dimensions_new, list(output_dimensions.keys()))

    output_dimensions_new = {}
    for dimname in dims_apply_names:
        for xarray in xarrays:
            if dimname in xarray.coords.keys():
                if dimname not in output_dimensions_new.keys():
                    logging.info('adding apply dimensions from input xarray to the default output_dimensions ' + dimname)
                    output_dimensions_new[dimname] = {'coords': xarray[dimname]}

    if output_dimensions is None:
        output_dimensions = {}

    for output_dimensions_orig in output_dimensions.keys():
        output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
    output_dimensions = sort_dict_by_keys(output_dimensions_new, list(output_dimensions.keys()))

    for dimname in dims_apply_names:
        if dimname not in output_dimensions:
            raise IOError('I cannot track the dimension size and coordinates of '+dimname+' please specify in output_dimensions')


        # for ixarray_output,xarray_output_dimensions in xarrays_output_dimensions:
        #     for dimname,coordinates in xarray_output_dimensions.items():
        #         if dimname not in output_dimensions.keys():
        #             output_dimensions[dimname] = xarray_output_dimensions[ixarray_output][dimname]




    number_of_chunks_apply = 1
    number_of_chunks_apply_dims = {}
    for dimname,dimattr in output_dimensions.items():
        if ('chunksize' in output_dimensions[dimname]):
            number_of_chunks_apply_dims[dimname] = int(np.ceil(len(output_dimensions[dimname]['coords'])/output_dimensions[dimname]['chunksize']))
            number_of_chunks_apply *= number_of_chunks_apply_dims[dimname] #np.ceil(len(output_dimensions[dimname]['coords'])/output_dimensions[dimname]['chunksize'])

    number_of_chunks_apply = int(number_of_chunks_apply)

    logging.info('Adding extra chunking as a separate dimension')
    if number_of_chunks_apply > 1:
        dims_no_apply['__chunk__'] = xr.DataArray(range(number_of_chunks_apply))

    dims_all = list(dims_no_apply.keys()) + dims_apply_names
    dims_no_apply_lengths = { name:dim.shape[0] for (name,dim) in dims_no_apply.items()}
    # dims_apply_shapes = { name:dim.shape for (name,dim) in dims_apply_names.items()}
    #dims_all_shapes = {**dims_no_apply_shapes,**dims_apply_shapes}


    # for dimname,coordinates in dims_no_apply.items():
    #     if dimname not in output_dimensions.keys():
    #         logging.info('adding no apply dimensions to the default output_dimensions '+dimname+': '+coordinates)
    #         output_dimensions[dimname] = {'coords':coordinates}

    while (len(xarrays_output_dimensions) < len(xarrays_output_filenames) ):
        logging.info('No coordinates output xarrays are set manually, so we guess them from the output_dimensions.'
                     'We do this here already so that we can take it into account in the memory size and optimal chunking.')
        xarrays_output_dimensions.append(output_dimensions)

    if len(xarrays_output_filenames) != len(xarrays_output_dimensions):
        raise IOError('number of output files are not the same as the number of expected output xarrays')

    output_coords = { key : (output_dimensions[key]['coords'] ) for key in output_dimensions.keys()}

    xarrays_output_coords = []
    for ixarray_out,xarray_output_dimensions in enumerate(xarrays_output_dimensions):
        xarrays_output_coords.append(
            {key: (xarray_output_dimensions[key]['coords'] ) for key in xarray_output_dimensions.keys()}
        )


    logging.info('Creating fake xarray outputs, '
                 'which we use to determine the no apply chunk sizes and the expected memory usage.')
    xarrays_out = []
    for ixarray_out in range(len(xarrays_output_coords)):
        shape = []
        dims = []
        for dim,coord in xarrays_output_coords[ixarray_out].items():
            if coord is not None:
                shape.append(len(coord))
                dims.append(dim)
        size = np.product(shape)
        nbytes = 4 * size
        xarrays_out.append(
            Namespace(**{
                           'coords':  xarrays_output_coords[ixarray_out],
                           'shape':shape,
                           'dims':dims,
                           'size': size,
                            'nbytes':nbytes}
                      )
        )


    xarrays_all = list(xarrays)+list(xarrays_out)
    xarrays_shapes_in_chunks = [list() for i in range(len(xarrays_all))]
    xarrays_shapes = [list() for i in range(len(xarrays_all))]
    #xarrays_shapes_in_chunks_cumulative = [1]*len(xarrays)
    #xarrays_dims_transposed = [[]]*len(xarrays)


    xarrays_chunks_apply = [False]*len(xarrays_all)
    for ixarray,xarray in enumerate(xarrays_all):
        for idim,dimname in reversed(list(enumerate(dims_apply_names))):
            #print(dimname,ixarray,xarrays_shapes_in_chunks)
            if dimname in xarray.dims:
                if (

                        (dimname in output_dimensions) and
                        #(output_dimensions[dimname] != None) and
                        ('chunksize' in list(output_dimensions[dimname].keys())) and
                        identical_xarrays(xarray.coords[dimname],output_dimensions[dimname]['coords'])
                   ):
                    xarrays_shapes_in_chunks[ixarray].insert(0,output_dimensions[dimname]['chunksize'])
                    if 'overlap' in output_dimensions[dimname]:
                        xarrays_shapes_in_chunks[ixarray][0] += output_dimensions[dimname]['overlap']

                    xarrays_chunks_apply[ixarray] = True
                else:
                    # if dimname in xarray.dims: # this case already caught above
                        xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray.coords[dimname]))
                        xarrays_shapes[ixarray].insert(0,xarray.shape[xarray.dims.index(dimname)])
                    # else:
                    #     xarrays_shapes_in_chunks[ixarray].insert(0, None)
                    #     xarrays_shapes[ixarray].insert(0,None)#xarray.shape[xarray.dims.index(dimname)])
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,None)
                xarrays_shapes[ixarray].insert(0,None)


    logging.info('adding size of __chunk__ dimensions')

    if number_of_chunks_apply > 1:
        for ixarray,xarray in enumerate(xarrays_all):
            if xarrays_chunks_apply[ixarray] == True:
                xarrays_shapes_in_chunks[ixarray].insert(0,number_of_chunks_apply)
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,None)
            xarrays_shapes[ixarray].insert(0,None)

    for idim,dimname in reversed(list(enumerate(dims_no_apply))):
        for ixarray,xarray in enumerate(xarrays_all):
            if dimname != '__chunk__': #inner extra chunk dimension is already considered in previous loop
                if dimname in xarray.dims:
                    xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray.coords[dimname]))
                    xarrays_shapes[ixarray].insert(0, len(xarray.coords[dimname]))
                else:
                    xarrays_shapes_in_chunks[ixarray].insert(0,None)
                    xarrays_shapes[ixarray].insert(0, None)


    logging.info('xarrays shapes for '+str(dims_no_apply.keys()) +' + '+str(dims_apply_names)+' : ')
    logging.info('  -> original xarrays: '+str(xarrays_shapes))
    logging.info('  ->  chunked xarrays: '+str(xarrays_shapes_in_chunks))
    logging.info('determining input chunk format that fits our maximum memory size input of '+str(maximum_memory_size_bytes))

    chunks_memory_sizes =     [int(xarray.nbytes/xarray.size) for xarray in xarrays_all]
    chunks_memory_sizes_dim = [[int(xarray.nbytes/xarray.size)] for xarray in xarrays_all]

    iteration_over_apply_dims = list(reversed(list(enumerate(dims_all))[-len(dims_apply_names):]))
    for idim,dimname in iteration_over_apply_dims:
        for ixarray,xarray in enumerate(xarrays_all):
            if xarrays_shapes_in_chunks[ixarray][idim] != None:
                chunks_memory_sizes[ixarray] *= xarrays_shapes_in_chunks[ixarray][idim]
                chunks_memory_sizes_dim[ixarray].insert(0,xarrays_shapes_in_chunks[ixarray][idim])

    iteration_over_noapply_dims = list(reversed(list(enumerate(dims_all))[:len(dims_no_apply)]))
    current_memory_size  = sum(chunks_memory_sizes)

    chunk_sizes_no_apply = {}
    for idim,dimname in iteration_over_noapply_dims:
        if (current_memory_size < maximum_memory_size_bytes):
            chunks_memory_sizes_total = sum(chunks_memory_sizes)
            xarrays_sized_cumulative_base = 0
            xarrays_sized_cumulative_mul = 0
            for ixarray,xarray in enumerate(xarrays_all):
                #xarrays_sizes_cumulative[ixarray] *= xarrays_shapes_in_chunks[ixarray][idim]
                if xarrays_shapes_in_chunks[ixarray][idim] == None:
                    xarrays_sized_cumulative_base += chunks_memory_sizes[ixarray]
                else:
                    xarrays_sized_cumulative_mul += chunks_memory_sizes[ixarray]

            if dimname == '__chunk__':
                logging.debug('We do not allow grouping of apply-dimension chunking.')
                chunk_sizes_no_apply[dimname] = 1.0
            else:
                if xarrays_sized_cumulative_mul  != 0:
                    chunk_sizes_no_apply[dimname] = np.floor(np.min([(maximum_memory_size_bytes - xarrays_sized_cumulative_base)/xarrays_sized_cumulative_mul, dims_no_apply_lengths[dimname]]))
                else:
                    chunk_sizes_no_apply[dimname] = 0

            if not chunk_sizes_no_apply[dimname].is_integer():
                raise ValueError('whole number expected for dimension selection size')

            current_memory_size = int((xarrays_sized_cumulative_base + xarrays_sized_cumulative_mul * chunk_sizes_no_apply[dimname]))
            #print(current_memory_size > maximum_memory_size)
            # if current_memory_size > maximum_memory_size:
            #     raise ValueError('something wrong with the chunk size calculation for limiting memory')
            for ixarray,xarray in enumerate(xarrays_all):
                if xarrays_shapes_in_chunks[ixarray][idim] != None:
                    chunks_memory_sizes[ixarray] *= chunk_sizes_no_apply[dimname]
                    chunks_memory_sizes_dim[ixarray].insert(0,chunk_sizes_no_apply[dimname])
                else:
                    chunks_memory_sizes_dim[ixarray].insert(0, None)
            if current_memory_size != sum(chunks_memory_sizes):
                import pdb; pdb.set_trace()
                raise ValueError('inconsistency in de dimension selection size calculation')
        else:
            for ixarray,xarray in enumerate(xarrays_all):
                chunks_memory_sizes_dim[ixarray].insert(0,1)
            chunk_sizes_no_apply[dimname] = 1

    logging.info('overall memory size: '+ str(current_memory_size))
    logging.info('xarray chunk memory sizes for arrays: '+ str(chunks_memory_sizes))
    logging.info('xarray chunk memory size per dimension (last one is character byte size): '+ str(chunks_memory_sizes_dim))

    chunks_number_no_apply = {}
    for dimname,shape in dims_no_apply_lengths.items():
        chunks_number_no_apply[dimname] = np.ceil(dims_no_apply_lengths[dimname]/chunk_sizes_no_apply[dimname])

    logging.info('memory input size of chunks: '+ str(current_memory_size) +'/'+ \
                 str(maximum_memory_size_bytes) +' = '+str(current_memory_size/int(maximum_memory_size_bytes)*100)+'% of maximum \n'+ \
                 ' - expected memory usage: ' + str(current_memory_size) + '\n' + \
                 ' - limit of memory usage: ' + str(current_memory_size) + '\n' + \
                 ' - chunks_memory_sizes: ' + str(chunks_memory_sizes) + '\n' + \
                 ' - chunks_memory_sizes_dim' + str(chunks_memory_sizes) + '\n' + \
                 'Please consider the usage of memory chunking along the apply_dimensions'
                 )
    if current_memory_size > maximum_memory_size_bytes:
        logging.warning('expected memory usage exceeds predefined memory limit!')
        if not ignore_memory_limit:
            raise IOError('memory limit needs to be respected. Or turn on ignore_memory_linit')

    #import pdb; pdb.set_trace()


    chunks_no_apply = list(product(*tuple([list(range(int(a))) for a in list(chunks_number_no_apply.values())])))


    first_chunks = True
    # logging.info('closing xarrays_out, which we used above to easily calculate the dimension/memory shaping. Data is written out directly through disk for saving memory. ')
    # for ixarray_out, xarray_out in enumerate(xarrays_out):

    #     xarrays_out[ixarray_out].close()

    # logging.info('Re-opening netcdf files for output
    # for ixarray_out, xarray_out in enumerate(xarrays_out):
    #     ncouts[ixarray_out] = nc4.Dataset(xarrays_output_filenames_work[ixarray_out],'a')


    logging.info('initialize lists for output netcdfs')
    ncouts = []
    xarrays_output_filenames_work = []
    xarrays_output_filenames_real = []
    xarrays_output_coords_final = []

    for index_no_apply in tqdm.tqdm(chunks_no_apply):
        logging.debug('dimension selection for '+\
                      str(dims_no_apply.keys)+\
                      ' with shape '+\
                      str(dims_no_apply_lengths.values())+\
                      ': '+str(index_no_apply))

        chunk_start = []
        chunk_end = []

        if '__chunk__' in chunks_number_no_apply.keys():
            dim_fac = 1
            idx_mod = index_no_apply[list(chunks_number_no_apply.keys()).index('__chunk__')]
        for dimname_apply in list(reversed(dims_apply_names)):
            if dimname_apply in number_of_chunks_apply_dims.keys():
                dim_apply_start = np.mod(idx_mod/dim_fac, number_of_chunks_apply_dims[dimname_apply] )
                chunk_start.insert(0,int(dim_apply_start * output_dimensions[dimname_apply]['chunksize']))
                chunk_end.insert(0,int(chunk_start[0] +  output_dimensions[dimname_apply]['chunksize']))
                if 'overlap' in output_dimensions[dimname_apply]:
                    chunk_end[0] += output_dimensions[dimname_apply]['overlap']
                chunk_end[0] = np.min([chunk_end[0],len(output_dimensions[dimname_apply]['coords'])])
                idx_mod -= dim_apply_start*dim_fac
                dim_fac *= number_of_chunks_apply_dims[dimname_apply]

                # chunk_start.insert(0,[])
                # chunk_end.insert(0,[])
                # for ichunk in range(int(chunks_number_no_apply[dimname])):
                #     dim_apply_start = np.mod(idx_mod/dim_fac, number_of_chunks_apply_dims[dimname_apply] )
                #     chunk_start[0].append(int(dim_apply_start * output_dimensions[dimname_apply]['chunksize']))
                #     chunk_end[0].append(int(chunk_start[0]+  output_dimensions[dimname_apply]['chunksize']))
                #     idx_mod -= dim_apply_start*dim_fac
                #     dim_fac *= number_of_chunks_apply_dims[dimname_apply]
            else:
                chunk_start.insert(0,None)
                chunk_end.insert(0,None)

        for idim in reversed(range(len(index_no_apply))):
            dimname = list(dims_no_apply_lengths.keys())[idim]
            if dimname != '__chunk__':
                chunk_start.insert(0,int(chunk_sizes_no_apply[dimname] * index_no_apply[idim]))
                chunk_end.insert(0,int(chunk_start[0] + chunk_sizes_no_apply[dimname]))
                chunk_end[0] = np.min([chunk_end[0],len(output_dimensions[dimname]['coords'])])
            else:
                chunk_start.insert(0,None)
                chunk_end.insert(0,None)

        xarrays_selection_chunk = []
        for ixarray, xarray in enumerate(xarrays_all):
            xarrays_selection_chunk.append({})
            for idim,dimname in enumerate(dims_all):
                #dimname = list(dims_no_apply_lengths.keys())[idim]
                if xarrays_shapes_in_chunks[ixarray][idim] is not None:
                    if dimname != '__chunk__':
                        if (dimname in dims_apply_names) and \
                                (dimname in output_dimensions) and \
                                ('chunksize' in output_dimensions[dimname]) and \
                                identical_xarrays(xarray.coords[dimname], output_dimensions[dimname]['coords']):
                            xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                        elif dimname in dims_no_apply.keys():
                            xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                        else:
                            xarrays_selection_chunk[ixarray][dimname] = range(0,xarrays_shapes_in_chunks[ixarray][idim])

        xarrays_selection_chunk_in = xarrays_selection_chunk[:len(xarrays)]
        xarrays_selection_chunk_out = xarrays_selection_chunk[len(xarrays):len(xarrays_all)]
        logging.debug('xarrays selection in  : '+str(xarrays_selection_chunk_in))
        logging.debug('xarrays selection out : '+str(xarrays_selection_chunk_out))

        chunks_in = []
        coordinates_in = []
        for ixarray,xarray in enumerate(xarrays):

            chunks_in.append(xarray.isel(xarrays_selection_chunk_in[ixarray]).transpose(*tuple(xarrays_selection_chunk_in[ixarray].keys())))

        dims_not_found = {}
        for dim,coordinate_output in output_dimensions.items():
            if output_dimensions[dim]['coords'] is not None:
                dimfound = False
                for ixarray,xarray in enumerate(xarrays):
                       if (dim in xarray.dims) and (coordinate_output['coords'] is not None) and (identical_xarrays(coordinate_output['coords'],xarray.coords[dim])):
                           dimfound = True

                if dimfound == False:
                    dims_not_found[dim] = coordinate_output['coords'].isel(
                        {dim:slice(chunk_start[dims_all.index(dim)], chunk_end[dims_all.index(dim)])}

                    )
        #chunk_output_dimensions = output_dimensions[dim].isel(art})
            if (len(dims_not_found) > 0):
                if (pass_missing_output_coordinates == True):
                    if (first_chunks == True):
                        logging.info('Output coordinates that are missing in the input files are found for '+str(dims_not_found.keys())+'. So we pass them to the function.')
                    pass_dims_not_found = dims_not_found
                else:
                    if (first_chunks == True):
                        logging.warning('Output coordinates that are missing in the input files are found for '+str(dims_not_found.keys())+". So the function doesn't know about it!")
                    pass_dims_not_found = {}
            else:
                pass_dims_not_found = {}


        # meshgrid_fine = np.meshgrid(
        #     chunks_in[1]['latitude'],
        #     chunks_in[1]['longitude'],
        #     indexing='ij')

        # func = lambda  x_coarse,**coordinates_fine: \
        #     (pcd.vectorized_functions.extend_crop_interpolate( \
        #     x_coarse.values, \
        #     (x_coarse.latitude.values,x_coarse.longitude.values), \
        #     (coordinates_fine['latitude'].values,coordinates_fine['longitude'].values), \
        #     # interpolation=True,
        #     # return_grid_output=False,
        #     # debug=False,
        #     # border_pixels=5,
        #     # ascending_lat_lon = False,
        #     # tolerance_for_grid_match = 1.e-9
        # ))

        chunks_out = func(*chunks_in,**pass_dims_not_found)

        # chunks_out = pcd.vectorized_functions.extend_crop_interpolate(
        #     chunks_in[0].values,
        #     (chunks_in[0].latitude.values, chunks_in[0].longitude.values,),
        #     (dims_not_found['latitude'].values, dims_not_found['longitude'].values),
        #     # interpolation=True,
        #     # return_grid_output=False,
        #     # debug=False,
        #     # border_pixels=5,
        #     # ascending_lat_lon = False,
        #     # tolerance_for_grid_match = 1.e-9
        # )
        if type(chunks_out).__name__ not in ['tuple', 'list']:
            # list_output = False
            chunks_out= [chunks_out]

        for ichunk_in in reversed(range(len(chunks_in))):
            chunks_in[ichunk_in].close()
            del chunks_in[ichunk_in]

        for ixarray_out,chunk_out in enumerate(chunks_out):
            logging.debug('xarray selection of chunk output '+str(ixarray_out)+': ' + str(xarrays_selection_chunk_out[ixarray_out]))

            xarrays_selection_chunk_out_ordered = sort_dict_by_keys(xarrays_selection_chunk_out[ixarray_out],list(xarrays_output_dimensions[ixarray_out].keys()))
            #if type(chunk_out) == type(np.array([])):
            if type(chunk_out) != xr.core.dataarray.DataArray:
                chunk_out_coordinates = {}
                for dimname in xarrays_selection_chunk_out[ixarray_out].keys():
                    chunk_out_coordinates[dimname] = xarrays_output_dimensions[ixarray_out][dimname]['coords'].isel({dimname:xarrays_selection_chunk_out[ixarray_out][dimname]})

                chunk_out_xarray = xr.DataArray(chunk_out,coords=chunk_out_coordinates)
            else:
                chunk_out_xarray = chunk_out

            logging.debug('xarray selection ordered for output array '+str(ixarray_out)+': ' + str(xarrays_selection_chunk_out_ordered))


            chunk_out_xarray_ordered = chunk_out_xarray.transpose(*tuple(xarrays_selection_chunk_out_ordered.keys()))
            # chunk_profile =

            logging.debug('re-ordered output shape: '+str(chunk_out_xarray.shape) +' -> '+ str(chunk_out_xarray_ordered.shape))
            indexing_for_output_array = tuple([dim_selection for dim_selection in xarrays_selection_chunk_out_ordered.values()])
            logging.debug('index of chunk in netcdf output '+str(ixarray_out)+': ' + str(indexing_for_output_array))
            #logging.debug('this should fit in netcdf total output shape '+str(ncouts[ixarray_out].variables['__xarray_data_variable__'].shape))
            logging.debug('this should fit in netcdf total output shape '+str(xarrays_out[ixarray_out].shape))

            overlap_weights = np.ones_like(chunk_out_xarray_ordered.values)
            idim = 0
            for dim,selection_chunk_out in xarrays_selection_chunk_out_ordered.items():
                # overlap_weights_dim = np.ones((len(xarrays_selection_chunk_out_ordered[dim],)))
                #reshape(list(range(idim-1))+overlap_weights.shape[idim])
                if 'overlap' in output_dimensions[dim]:
                    if xarrays_selection_chunk_out_ordered[dim][0] == 0:
                        left = np.ones(output_dimensions[dim]['overlap'])
                    else:
                        left = np.arange(0,output_dimensions[dim]['overlap'],1)/output_dimensions[dim]['overlap']
                    if xarrays_selection_chunk_out_ordered[dim][-1] == (len(output_dimensions[dim]['coords']) - 1):
                        right = np.ones(output_dimensions[dim]['overlap'])
                    else:
                        right = np.arange(output_dimensions[dim]['overlap'],0,-1)/output_dimensions[dim]['overlap']
                    middle = np.ones(((len(xarrays_selection_chunk_out_ordered[dim]) - 2 * output_dimensions[dim]['overlap'],)))
                    overlap_weights_dim = np.concatenate([left,middle,right])
                    overlap_weights_dim = overlap_weights_dim.reshape([1]*idim+[overlap_weights.shape[idim]]+[1]*(len(overlap_weights.shape) - idim -1))
                    overlap_weights *= overlap_weights_dim
                idim += 1


            if first_chunks == True:
                attributes_out = {}
                logging.info('propagate attributes from xarray chunk function output')
                for attrkey,attrvalue in chunk_out_xarray.attrs.items():
                    attributes_out[attrkey] = attrvalue

                logging.info('update attributes derived from possible new coordinate system')
                # xarray_out = xr.open_dataarray(xarrays_output_filenames_work)
                # coordinates_attributes = get_coordinates_attributes(xarrays_output_coords[incout])
                coordinates_attributes = get_coordinates_attributes(xarrays_output_coords[ixarray_out])
                for dim in dims_apply_names:
                    # if coordinates_attributes[dim] is None:
                    #     import pdb; pdb.set_trace()
                    #     if dim not in attributes_out.keys():
                    #         coordinates_attributes[dim] = 'None'
                        if ((dim in xarrays_out[ixarray_out].dims) and not identical_xarrays(xarrays_out[ixarray_out].coords[dim],xarrays_output_coords[ixarray_out][dim])) and \
                                (dim in coordinates_attributes.keys()):
                            attributes_out[dim] = coordinates_attributes[dim]

                if ('latitude' in dims_apply_names) or ('longitude' in dims_apply_names):
                    if (( 'latitude' in xarrays[0].dims) and (not identical_xarrays(xarrays_output_coords[ixarray_out]['latitude'],xarrays[0][dim]))) and \
                       (( 'longitude' in xarrays[0].dims) and ( not identical_xarrays(xarrays_output_coords[ixarray_out]['longitude'], xarrays[0][dim]))) and \
                       ('space' in coordinates_attributes.keys()):
                        attributes_out['space'] = coordinates_attributes['space']


                xarrays_output_coords_final.append({})
                dims = []
                shape = []
                for dim in chunk_out_xarray_ordered.dims:
                    if (dim not in dims_apply_names) or \
                            ((dim in output_dimensions.keys()) and ('chunksize' in output_dimensions[dim])):
                        xarrays_output_coords_final[ixarray_out][dim] = output_coords[dim]
                    else:
                        xarrays_output_coords_final[ixarray_out][dim] = chunk_out_xarray_ordered.coords[dim]
                    shape.append(len(xarrays_output_coords_final[ixarray_out][dim]))
                    dims.append(dim)

                logging.info('checking consistency of expected output format between function output chunk and expected array output.')

                if ixarray_out >= len(xarrays_out):
                    IOError('Function output has at least '+str(ixarray_output+1)+' output xarrays, but only '+str(len(xarrays_output)+' is/are expected.'))

                if dims != xarrays_out[ixarray_out].dims:
                    IOError('Function output chunk has different dimensions than expected output dimensions.')
                if shape != xarrays_out[ixarray_out].shape:
                    IOError('Function output chunk has different shape than expected output dimensions.')

                logging.info('acquiring real output filename for xarray out number '+str(ixarray_out)+' and setting output (temporary filename)')
                xarrays_output_filenames_real.append(
                    name_from_pattern(
                        xarrays_output_filenames[ixarray_out],
                        {**attributes_out,**{'variable':chunk_out_xarray_ordered.name}}
                    )
                )

                if os.path.isfile(xarrays_output_filenames_real[ixarray_out]):
                    if overwrite_output_filenames == False:
                        raise FileExistsError(
                            xarrays_output_filenames_real[ixarray_out] + ' ( ' + xarrays_output_filenames[ixarray_out] + ' ) exists.'
                        )
                    else:
                       logging.warning(
                           'Filename output ' + xarrays_output_filenames_real[ixarray_out] + '(' + \
                           xarrays_output_filenames[ ixarray_out] + ') exists. Removing before writing.'
                       )
                       os.system('rm '+xarrays_output_filenames_real[ixarray_out])

                if not tempfile_dir:
                    xarrays_output_filenames_work.append(xarrays_output_filenames_real[ixarray_out])
                    logging.info("Dump output directly to final destination: " + xarrays_output_filenames_work[-1])
                else:
                    logging.info("Using temporary output dir, eg., good for working with network file systems")
                    if (tempfile_dir is None) or (tempfile_dir is True):
                        xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=None))
                        logging.info("Using temporary output in default tempfile_dir: " + xarrays_output_filenames_work[-1])
                    else:
                        xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=tempfile_dir))
                        logging.info("Using temporary output in specified tempfile_dir: " + xarrays_output_filenames_work[-1])

                xrtemp = xr.Dataset()
                #for ixarray_out in range(len(xarrays_output_dimensions)):
                for dimname, coords in xarrays_output_coords_final[ixarray_out].items():
                    xrtemp[dimname] = coords
                    # ncouts[iarray].createDimension(dim,shapes_out_transposed[iarray][idim])
                    # ncouts[iarray].createVariable(dim,'d',(dim,),)
                    # ncouts[iarray].variables[dim][:] = coords_out_transposed[iarray][idim]
                fnout = xarrays_output_filenames_work[ixarray_out]  # 'testing_'+str(iarray)+'.nc'
                if os.path.isfile(fnout):
                    raise FileExistsError('output file ' + fnout + ' exists. Aborting... ')
                # os.system('rm ' + fnout)
                xrtemp.to_netcdf(fnout)
                xrtemp.close()
                logging.info('creating netcdf file '+fnout)
                ncouts.append(nc4.Dataset(fnout, 'a'))

                try:
                    ncouts[ixarray_out].createVariable(chunk_out_xarray_ordered.name, "f", tuple(xarrays_output_coords_final[ixarray_out].keys()),fill_value=0.)
                #ncouts[ixarray_out].variables['__xarray_data_variable__'][:] = 0.
                except:
                    import pdb; pdb.set_trace()

                logging.info('setting netcdf variable attributes: '+str(attributes_out))

                def fix_dict_for_ncattributes(attributes):
                    attributes_out = {}
                    for attrkey,attrvalue in attributes.items():
                        if (type(attrvalue) == str) and ( attrvalue == ''):
                            logging.warning('Excluding attribute "'+attrkey+'" that has empty value. Apparently, this gives problems when writing to the netcdf later on.')
                        else:
                            attributes_out[attrkey] = attrvalue
                    return attributes_out
                attributes_out = fix_dict_for_ncattributes(attributes_out)

                for attrkey,attrvalue in attributes_out.items():
                    ncouts[ixarray_out].variables[chunk_out_xarray_ordered.name].setncattr(attrkey,attrvalue)
                #xarrays_out.append(xr.open_dataarray(fnout))
                logging.info('finished initializing netcdf file '+str(ixarray_out))


            logging.debug('acquiring previous values for consolidating chunk overlapping values')
            # try:
            test = ncouts[ixarray_out].variables[chunk_out_xarray_ordered.name][indexing_for_output_array].filled(fill_value=0)
            #   except:
            #       import pdb; pdb.set_trace()

            logging.debug('writing chunk ('+str(indexing_for_output_array)+') to netcdf file '+str(ixarray_out))
            if first_chunks:
                logging.info('writing first chunk ('+str(indexing_for_output_array)+') to netcdf file '+str(ixarray_out)+'. This takes a much longer than the next chunks because of some hidden initializations of the netcdf file.')
            ncouts[ixarray_out].variables[chunk_out_xarray_ordered.name][indexing_for_output_array] = test + np.array( chunk_out_xarray_ordered.values, dtype='float32') * overlap_weights
            if first_chunks:
                logging.info('finished writing first chunk')
            del chunk_out_xarray
            del chunk_out_xarray_ordered
            logging.debug('.... finished')

        for ichunk_out in range(len(chunks_out)):
            if type(chunks_out[ichunk_out]) == xr.core.dataarray.DataArray:
                chunks_out[ichunk_out].close()

            #del chunks_out[ichunk_out]
            # import pdb; pdb.set_trace()
        first_chunks = False

    for incout in range(len(ncouts)):
        # #xarray_out = xr.open_dataarray(xarrays_output_filenames_work)
        # #coordinates_attributes = get_coordinates_attributes(xarrays_output_coords[incout])
        # coordinates_attributes = get_coordinates_attributes(xarrays_output_coords[incout])
        # for dim in dims_apply_names:
        #     if (not identical_xarrays(xarrays_output_coords[incout][dim] )) and \
        #             (dim in coordinates_attributes.keys()):
        #         ncout[incout]['__xarray_variable__'].set_attr(dim, coordinates_attributes)

        # if ('latitude' in dims_apply_names) or ('longitude' in dims_apply_names):
        #     if (not identical_xarrays(xarrays_output_coords[incout]['latitude'] )) and \
        #        (not identical_xarrays(xarrays_output_coords[incout]['longitude'])) and \
        #        ('space' in coordinates_attributes.keys()):
        #         ncout[incout]['__xarray_variable__'].set_attr(dim, coordinates_attributes)

        ncouts[incout].close()

        # ?????
        #archive_out.lib_dataarrays.apply(lambda x: os.path.realpath(os.path_realpath(os.path.dirname(x['path_pickle']))+'/'+x['path']),axis='columns')
        # import pdb; pdb.set_trace()
        # if (os.path.realpath(filename_out) in absolute_paths) and (not dataarrays_out_already_available[ifile]):
        #     raise ValueError(
        #         'filename ' + filename_out + ' already exists and not already managed/within the output archive. Consider revising the output file_pattern.')

        CMD = 'mv '+xarrays_output_filenames_work[incout]+' '+xarrays_output_filenames_real[incout]
        logging.info('Moving temporary output to actual netcdf: '+CMD)
        os.system(CMD)
    return xarrays_output_filenames_real


if __name__ == '__main__':
    input_file = '/home/woutersh/projects/KLIMPALA_SF/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
    ds = xr.open_dataset( input_file,)['Band1'].rename({'lat':'latitude'}).rename({'lon':'longitude'})
    ds = ds.isel(
        latitude  = ( (ds.latitude  > -5) & (ds.latitude  < 5)),
        longitude = ( (ds.longitude > -10) & (ds.longitude < 5))
    )

    #input_file = '/projects/C3S_EUBiodiversity/data/case_klimpala/aggregation-30-years/indicators-annual/cropped_to_africa/bias_corrected/cmip5_daily/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
    input_file = '/home/woutersh/projects/KLIMPALA_SF/data/test/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
    ds2 = xr.open_dataarray(input_file)

    # this also sets the order of (inner) output dimensions as expected by the function
    dims_apply_names = ['latitude','longitude']

    ignore_memory_limit = False

    # this also defines the order in which the output dimensions are being constructed, allowing for transposing the output on the fly.
    output_dimensions = { #input
        #        'time':{ 'coords':ds2.time},
        'longitude':{ 'coords':ds.longitude,'chunksize':1000,'overlap':50},
        'latitude':{ 'coords':ds.latitude,'chunksize':500,'overlap':50},
    }
    #xarrays = [ds2,ds.latitude,ds.longitude]
    xarrays = [ds2,ds.latitude,ds.longitude]

    xarrays_output_dimensions = [] #default
    #xarrays_output_filenames =

    def func(x_coarse, latitude,longitude):
        out = (pcd.vectorized_functions.extend_crop_interpolate( \
            x_coarse.values, \
            (x_coarse.latitude.values, x_coarse.longitude.values), \
            (latitude.values, longitude.values), \
            # interpolation=True,
            # return_grid_output=False,
            # debug=False,
            # border_pixels=5,
            # ascending_lat_lon = False,
            # tolerance_for_grid_match = 1.e-9
        ))
        coords = dict(x_coarse.coords)
        coords['latitude'] =  latitude
        coords['longitude'] =  longitude
        xrout = xr.DataArray(out,dims=x_coarse.dims,coords=coords)
        xrout.name = x_coarse.name
        xrout.attrs = x_coarse.attrs
        return xrout, xrout
    apply_func(
        func,
        xarrays,
        dims_apply_names = ['latitude','longitude'],
        xarrays_output_filenames = [
            '/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing.nc',
            '/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing2.nc'],
        #attributes = None,
        output_dimensions=output_dimensions,
        maximum_memory_size_bytes=2 * 10 ** 7,
    #squeeze_apply_dims = False,
        tempfile_dir='/tmp/',
        overwrite_output_filenames=True,
        pass_missing_output_coordinates=False,
    )
