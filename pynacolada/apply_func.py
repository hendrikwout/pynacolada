import dask.array as da
from functools import partial
from multiprocessing.pool import Pool
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
from time import sleep
import tqdm
import logging
import sys
import datetime as dt
sys.path.insert(0, 'lib/pynacolada/')
import pynacolada as pcd

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


def xarray_coordinate_from_array(array, name):
    return xr.DataArray(array, dims=[name], name=name, coords={name: array})

def get_dimensions_attributes(coords,time_id=None,space_id=None):#,prepend={},append={}):

    dimensions_attributes = {}
    if 'time' in coords.keys():
        # if ('time' not in dict_index.keys()) or (dict_index['time'] is None) or (
        #         type(dict_index['time']).__name__ == 'float') and (np.isnan(dict_index['time']).any()):
        #     print('Guessing time coordinate from DataArray')
        #     # is month type

        # monthly spacing
        if coords['time'] is None:
            dimensions_attributes['time'] = None
        elif str(coords['time'].dtype).startswith('int') :
            dimensions_attributes['time'] = 'integer_'+str(coords['time'].values[0])+'_'+str(coords['time'].values[-1])
        elif np.apply_along_axis(lambda y: np.sum((y[1:] - y[:-1] != 1), 0), 0,
                               np.vectorize(lambda x: int(x[:4]) * 12 + int(x[5:7]))(
                                   coords['time'].values.astype('str'))).item() == 0:
            dimensions_attributes['time'] = \
                'monthly_' + str(coords['time'][0].values)[:7] + '_' + str(coords['time'][-1].values)[:7]
        # also monthly
        elif (not np.any(~(np.vectorize(lambda x: x[8:])(
                coords['time'].values.astype('str')) == '01T00:00:00.000000000'))):
            dimensions_attributes['time'] = \
                'monthly_' + str(coords['time'][0].values)[:7] + '_' + str(coords['time'][-1].values)[:7]
        elif not np.any((coords['time'][2:-1].values - coords['time'][1:-2].values) != np.array(86400000000000,
                                                                                                dtype='timedelta64[ns]')):
            # 
            dimensions_attributes['time'] = '_' + np.datetime_as_string(coords['time'][0].values,
                                                                              unit='D') + '_' + np.datetime_as_string(
                coords['time'][-1].values, unit='D')
        elif not np.any((coords['time'][2:-1].values - coords['time'][1:-2].values) != dt.timedelta(days=1)):
            dimensions_attributes['time'] = \
                'daily_' + str(coords['time'][0].values)[:10] + '_' + str(coords['time'][-1].values)[:10]
        else:
            dimensions_attributes['time'] = 'irregular'
            logging.warning('No time dimension found')
        if time_id != None:
            dimensions_attributes['time'] = time_id + '_'+ dimensions_attributes['time']


        # if 'time' in prepend.keys():
        #     dimensions_attributes['time'] = prepend['time']+dimensions_attributes['time']
        # if 'time' in append.keys():
        #     dimensions_attributes['time'] = dimensions_attributes['time']+append['time']

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
    # space_coordinates = ['latitude','longitude']

    spacing = {}
    for dim,coord in coords.items():
        if (dim != 'time') and (coord is not None):# space_coordinates:
            #import pdb; pdb.set_trace()
            spacing_temp = (coords[dim].values[1] - coord[dim].values[0])
            if not np.any(
                    coords[dim][1:].values != (coords[dim].values[:-1] + spacing_temp)):
                spacing[dim] = str(coords[dim][0].values) + ',' + str(
                    coords[dim][-1].values) + ',' + str(spacing_temp)
            else:
                spacing[dim] = 'irregular'
            # else:
            #     logging.warning('unknown dimension found that we will not be tracked in lib_dataarrays: ' + str(dim))
    dict_index_space = [key + ':' + str(value) for key, value in spacing.items()]
    if len(dict_index_space) != 0:
        space_label = '_'.join(dict_index_space)
    else:
        space_label = None
        #space_label = '_no_space_'

    if space_id != None:
        space_label = space_id+'_'+space_label

    dimensions_attributes['space'] = space_label

    return dimensions_attributes


def identical_xarrays(xarray1,xarray2):
    return \
        (not ( ( (xarray1 is not None) and (xarray2 is None)) or ( (xarray1 is None) and (xarray2 is not None)))) and \
        (xarray1.shape == xarray2.shape) and \
        (not (np.any(xarray1.values != xarray2.values)))

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

barposition = 0


def get_xarrays_selection_chunk(
        xarrays,
        xarrays_shapes_chunks,
        dims_all,
        output_dimensions,
        dims_apply_names,
        number_of_chunks_apply_dims,
        dims_no_apply,
        chunk_start,
        chunk_end,
):
    xarrays_selection_chunk = []
    for ixarray, xarray in enumerate(xarrays):
        xarrays_selection_chunk.append({})
        for idim, dimname in enumerate(dims_all):
            # dimname = list(dims_no_apply_lengths.keys())[idim]
            if xarrays_shapes_chunks[ixarray][idim] is not None:
                # if (ixarray == 1) and (dimname == 'latitude'):
                if dimname != '__chunk__':
                    if (dimname in dims_apply_names) and \
                            (dimname in output_dimensions) and \
                            ('chunksize' in output_dimensions[dimname]) and \
                            (dimname in number_of_chunks_apply_dims.keys()) and \
                            identical_xarrays(xarray.coords[dimname], output_dimensions[dimname]['coords']):
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim], chunk_end[idim])
                    elif dimname in dims_apply_names:
                        # xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim], chunk_end[idim])
                        if (dimname in xarray.dims):
                            if (xarrays_shapes_chunks[ixarray][idim] == 1):
                                xarrays_selection_chunk[ixarray][dimname] = range(0, 1)
                            else:
                                xarrays_selection_chunk[ixarray][dimname] = range(0, xarrays_shapes_chunks[ixarray][
                                    idim])
                    elif dimname in dims_no_apply.keys():
                        if (dimname in xarray.dims):
                            if (xarrays_shapes_chunks[ixarray][idim] == 1):
                                xarrays_selection_chunk[ixarray][dimname] = range(0, 1)
                            else:
                                xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],
                                                                                  chunk_end[idim])
    return xarrays_selection_chunk


def chunk_task(func,
               chunks_number_no_apply,
               dims_apply_names,
               number_of_chunks_apply_dims,
               output_dimensions,
               dims_no_apply_lengths,
               chunk_sizes_no_apply,
               xarrays_in,
               xarrays_in_shapes_chunks,
               dims_all,
               dims_no_apply,
               pass_missing_output_coordinates,
               args_func=[],
               kwargs_func={},
               index_no_apply=None,
               ):

    # logging.debug('dimension selection for ' + \
    #               str(dims_no_apply.keys) + \
    #               ' with shape ' + \
    #               str(dims_no_apply_lengths.values()) + \
    #               ': ' + str(index_no_apply))

    # logging.debug('determining chunk start end ranges along different dimensions.')

    chunk_start = []
    chunk_end = []

    if '__chunk__' in chunks_number_no_apply.keys():
        dim_fac = 1
        idx_mod = index_no_apply[list(chunks_number_no_apply.keys()).index('__chunk__')]
    for dimname_apply in list(reversed(dims_apply_names)):
        if dimname_apply in number_of_chunks_apply_dims.keys():
            dim_apply_start = np.mod(idx_mod / dim_fac, number_of_chunks_apply_dims[dimname_apply])
            if 'overlap' in output_dimensions[dimname_apply]:
                chunk_start.insert(0,
                                   int(dim_apply_start * (
                                           output_dimensions[dimname_apply]['chunksize'] -
                                           output_dimensions[dimname_apply]['overlap'])))
            else:
                chunk_start.insert(0,
                                   int(dim_apply_start * (output_dimensions[dimname_apply]['chunksize'])))

            chunk_end.insert(0, int(chunk_start[0] + output_dimensions[dimname_apply]['chunksize']))
            # if 'overlap' in output_dimensions[dimname_apply]:
            #    chunk_end[0] += output_dimensions[dimname_apply]['overlap']

            chunk_end[0] = np.min([chunk_end[0], len(output_dimensions[dimname_apply]['coords'])])
            #chunk_start[0] = chunk_end[0] - output_dimensions[dimname_apply]['chunksize']

            idx_mod -= dim_apply_start * dim_fac
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
            chunk_start.insert(0, None)
            chunk_end.insert(0, None)

    for idim in reversed(range(len(index_no_apply))):
        dimname = list(dims_no_apply_lengths.keys())[idim]
        if dimname != '__chunk__':
            chunk_start.insert(0, int(chunk_sizes_no_apply[dimname] * index_no_apply[idim]))
            chunk_end.insert(0, int(chunk_start[0] + chunk_sizes_no_apply[dimname]))
            chunk_end[0] = np.min([chunk_end[0], len(output_dimensions[dimname]['coords'])])
        else:
            chunk_start.insert(0, None)
            chunk_end.insert(0, None)

    # logging.info('create selection xarrays input from chunk_start (' + \
    #              str(chunk_start) + ') and chunk_end (' + str(chunk_end) + ').')

    xarrays_in_selection_chunk = get_xarrays_selection_chunk(
        xarrays_in,
        xarrays_in_shapes_chunks,  # this also includes the apply_chunk dimension
        dims_all,  # dimensions in the order as seen by the function/operator
        output_dimensions,  # dimensions in the order for the output
        dims_apply_names,
        number_of_chunks_apply_dims,
        dims_no_apply,
        chunk_start,
        chunk_end,
    )

    chunks_in = []
    for ixarray, xarray in enumerate(xarrays_in):

        chunks_in.append(xarray.isel(xarrays_in_selection_chunk[ixarray]).transpose(
            *tuple(xarrays_in_selection_chunk[ixarray].keys())))

        if 'variable' in chunks_in[-1].attrs.keys():
    #        logging.info('variable attribute detected. Replacing xarray variable name to this value.')
            chunks_in[-1].name = str(chunks_in[-1].attrs['variable'])
            del chunks_in[-1].attrs['variable']

    dims_not_found = {}
    for dim, coordinate_output in output_dimensions.items():
        if output_dimensions[dim]['coords'] is not None:
            dimfound = False
            for ixarray, xarray in enumerate(xarrays_in):
                if (dim in xarray.dims) and (
                        identical_xarrays(coordinate_output['coords'], xarray.coords[dim])):
                    dimfound = True

            if dimfound == False:
                dims_not_found[dim] = coordinate_output['coords'].isel(
                    {dim: slice(chunk_start[dims_all.index(dim)], chunk_end[dims_all.index(dim)])}

                )

    if (len(dims_not_found) > 0):
        if (pass_missing_output_coordinates == True):
            # if (first_chunks == True):
            #     logging.info('Output coordinates that are missing in the input files are found for ' + str(
            #         dims_not_found.keys()) + '. So we pass them to the function.')
            pass_dims_not_found = dims_not_found
        else:
            # if (first_chunks == True):
            #     logging.warning(
            #         'Output coordinates that are missing in the input files are found for ' + str(
            #             dims_not_found.keys()) + ". So the function doesn't know about it! . Use pass_missing_output_coordinates = True to pass them to the function.")
            pass_dims_not_found = {}
    else:
        pass_dims_not_found = {}

    chunks_out = func(*chunks_in, *args_func,**pass_dims_not_found,**kwargs_func)
    if type(chunks_out) not in [list, tuple]:
        chunks_out = [chunks_out]

    chunks_out_xarrays = []

    for ixarray_out, chunk_out in enumerate(chunks_out):
        if type(chunk_out) != xr.DataArray:
            logging.info('Numpy output detected. Converting to xarray and guess/assign coordinate attributes according to input. ')
            chunk_out_coordinates = []
            for idim,dimlength in list(reversed(list(enumerate(chunk_out.shape)))):
                ##### WARNING ... this will give error... treating non-xarray output needs to be fixed
                # if dimlength == 1:
                #     chunk_out_coordinates[dims_all[idim]] = np.array[0]
                # else:
                
                chunk_out_coordinates.insert(0,[dims_all[idim], None])
                for ichunk_in,chunk_in in reversed(list(enumerate(chunks_in))): 
                    if (dims_all[idim] in chunk_in.dims):
                        if (len(chunk_in[dims_all[idim]]) == dimlength):
                            chunk_out_coordinates[0] = [dims_all[idim] , chunk_in[dims_all[idim]]]
                        elif dimlength == 1:
                            middleco = int(len(chunk_in[dims_all[idim]])/2)
                            chunk_out_coordinates[0] = [dims_all[idim], chunk_in[dims_all[idim]][middleco:middleco+1]]


                if chunk_out_coordinates[0] is None: 
                    chunk_out_coordinates[0] = np.arange(dimlength)
            chunk_out_coordinates = dict(chunk_out_coordinates) 


            logging.debug('for dimensions over which function is applied, we select coordinates of the chunk')
            # perhaps we should do the same for dims_no_apply, instead of the code just above!
            for idim,dim in enumerate(dims_apply_names):
                chunk_out_coordinates_dim = output_dimensions[dim]['coords']
                if chunk_end[-1-idim] is not None:
                    chunk_out_coordinates_dim = chunk_out_coordinates_dim[:chunk_end[-1-idim]]
                if chunk_start[-1-idim] is not None:
                    chunk_out_coordinates_dim = chunk_out_coordinates_dim[chunk_start[-1-idim]:]

                chunk_out_coordinates[dim] = chunk_out_coordinates_dim

            chunk_out_xarray = xr.DataArray(chunk_out, dims=chunk_out_coordinates.keys(),coords=chunk_out_coordinates)

            #chunk_out_xarray.name = chunks_in[0].name
            
            # for ichunk_in,chunk_in in reversed(list(enumerate(chunks_in))): 
            #     for attrkey,attrvalue in chunk_in.attrs.items():
            #         chunk_out_xarray.attrs[attrkey] = attrvalue

            chunks_out_xarrays.append(chunk_out_xarray)
        else:
            chunks_out_xarrays.append(chunk_out)

    for ichunk_in in reversed(range(len(chunks_in))):
        chunks_in[ichunk_in].close()
        del chunks_in[ichunk_in]
    return chunks_out_xarrays, chunk_start, chunk_end


def apply_func(
        func,
        xarrays_in,
        dims_apply_names = [],
        xarrays_output_filenames = False,
        attributes = None,
        maximum_memory_size_per_proc_bytes = 2 * 10 ** 8 ,
        output_dimensions=None,
        xarrays_output_dimensions = None,
        tempfile_dir=False,
        return_type='xarrays',
        ignore_memory_limit = False,
        overwrite_output_filenames = True,
        pass_missing_output_coordinates = False,
        profile_overlap = 'square',
        nprocs = 1,
        args_func = [],
        kwargs_func = {},
):

    global barposition
    barposition = barposition+1
    #maximum_memory_size_bytes_per_proc = int(maximum_memory_size_bytes/nprocs)

    for xarray_in in xarrays_in:
        for lendim in xarray_in.shape:
            if lendim == 0:
                raise ValueError('xarrays with dimension length 0 not yet supported.')

    #input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'

    # xarrays_output_dimensions = [ #input
    #     {
    #         'time': {'coords': ds2.time},
    #         'latitude': {'coords': ds.latitude, 'chunksize': 4500},
    #         'longitude': {'coords': ds.longitude, 'chunksize': 4500},
    #     }
    # ]

    logging.debug('collecting the dimensions occuring in the xarrays over which the function is not applied')
    dims_no_apply  = {}

    for ixarray_in,xarray_in in enumerate(xarrays_in):

        if type(xarray_in) == xr.DataArray:
            for dimname in xarray_in.dims:
                if (dimname not in dims_no_apply) and (dimname not in dims_apply_names):
                    dims_no_apply[dimname] = xarray_in[dimname]
                if (dimname in dims_no_apply) and (dimname in xarray_in.dims) and (not identical_xarrays(xarray_in[dimname],dims_no_apply[dimname])) and (len(xarray_in[dimname]) != 1):
                    import pdb; pdb.set_trace()
                    raise ValueError('dimension '+dimname+' of xarray_in number '+str(ixarray_in)+' is not the same as previously detected dimensions.')
        else:
            raise IOError('input array type not supported.')


    # if output_dimensions is None:
    #     output_dimensions_sort_keys = list(output_dimensions.keys())
    # else:
    #     output_dimensions_sort_keys = []

    logging.debug('adding missing output dimensions from specified xarrays_coordinates_output')
    if xarrays_output_dimensions is not None:
        output_dimensions_new = {}
        for dimdict in xarrays_output_dimensions:
            for dimname in dimdict.keys():
                 if dimname not in output_dimensions_new.keys():
                     logging.debug('adding missing output_dimensions from xarrays_output_dimensions for '+dimname+': ')#+dimdict[dimname])
                     output_dimensions_new[dimname] = dimdict[dimname]
        logging.debug('overriding values and order of previous output_dimensions.')

        if output_dimensions is None:
            output_dimensions = dict()

        for output_dimensions_orig in output_dimensions.keys():
            output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
        output_dimensions = sort_dict_by_keys(output_dimensions_new,list(output_dimensions.keys()))

    output_dimensions_new = {}

    for xarray in xarrays_in:
        for dimname in xarray.dims:
            if dimname in list(dims_no_apply.keys())+dims_apply_names:
                if dimname not in output_dimensions_new.keys():
                    logging.debug('adding apply dimensions from input xarray to the default output_dimensions ' + dimname)
                    output_dimensions_new[dimname] = {'coords': xarray[dimname]}

    if output_dimensions is None:
        output_dimensions = {}

    for output_dimensions_orig in output_dimensions.keys():
        output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
    output_dimensions = sort_dict_by_keys(output_dimensions_new, list(output_dimensions.keys()))

    output_dimensions_new = {}
    for dimname,coordinates in dims_no_apply.items():

        if dimname not in output_dimensions_new.keys():
            logging.debug('adding no apply dimensions to the default output_dimensions '+dimname)
            output_dimensions_new[dimname] = {'coords':coordinates}

    if output_dimensions is None:
        output_dimensions = {}

    for output_dimensions_orig in output_dimensions.keys():
        output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
    output_dimensions = sort_dict_by_keys(output_dimensions_new, list(output_dimensions.keys()))

    # for dimname in dims_apply_names:
    #     if dimname not in output_dimensions:
    #         raise IOError('I cannot track the dimension size and coordinates of '+dimname+' please specify in output_dimensions')

    output_dimensions_new = {}
    for dimname in dims_apply_names:
        if dimname not in output_dimensions.keys():
            logging.debug('adding no apply dimensions to the default output_dimensions '+dimname)
            output_dimensions_new[dimname] = {'coords':None}

    if output_dimensions is None:
        output_dimensions = {}

    for output_dimensions_orig in output_dimensions.keys():
        output_dimensions_new[output_dimensions_orig] = output_dimensions[output_dimensions_orig]
    output_dimensions = sort_dict_by_keys(output_dimensions_new, list(output_dimensions.keys()))


        # for ixarray_output,xarray_output_dimensions in xarrays_output_dimensions:
        #     for dimname,coordinates in xarray_output_dimensions.items():
        #         if dimname not in output_dimensions.keys():
        #             output_dimensions[dimname] = xarray_output_dimensions[ixarray_output][dimname]




    number_of_chunks_apply = 1
    number_of_chunks_apply_dims = {}
    for dimname,dimattr in output_dimensions.items():

        if (output_dimensions[dimname] != None) and ('chunksize' in output_dimensions[dimname]):
            if 'overlap' in output_dimensions[dimname]:
                number_of_chunks_apply_dims_dimname = int(
                    np.ceil(len(output_dimensions[dimname]['coords']) / (
                            output_dimensions[dimname]['chunksize'] - output_dimensions[dimname]['overlap'])))
            else:
                number_of_chunks_apply_dims_dimname = int(
                    np.ceil(len(output_dimensions[dimname]['coords']) / output_dimensions[dimname]['chunksize']))
            if number_of_chunks_apply_dims_dimname > 1:
                number_of_chunks_apply_dims[dimname] = number_of_chunks_apply_dims_dimname
                number_of_chunks_apply *= number_of_chunks_apply_dims[dimname] #np.ceil(len(output_dimensions[dimname]['coords'])/output_dimensions[dimname]['chunksize'])

    number_of_chunks_apply = int(number_of_chunks_apply)

    logging.debug('Adding extra chunking as a separate dimension')
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

    if xarrays_output_filenames == None:
        xarrays_output_filenames = tempfile.mktemp(suffix='.nc', dir='/tmp/' )

    if xarrays_output_dimensions is None:
        if (xarrays_output_filenames == None) or (xarrays_output_filenames == False) or (type(xarrays_output_filenames) == str):
            logging.debug('by default, we assume only one xarray output.')
            xarrays_output_dimensions = [output_dimensions]

        elif type(xarrays_output_filenames) in (list,tuple):
            xarrays_output_dimensions = list()
            while (len(xarrays_output_dimensions) < len(xarrays_output_filenames) ):
                logging.info('No coordinates output xarrays are set manually, so we guess them from the output_dimensions.'
                             'We do this here already so that we can take it into account in the memory size and optimal chunking.')
                xarrays_output_dimensions.append(output_dimensions)
        else:
            raise ValueError('xarrays_output_filenames should be string or list.')

    if (xarrays_output_filenames != False) and (xarrays_output_filenames != None) and (type(xarrays_output_filenames) != str) and (len(xarrays_output_filenames) != len(xarrays_output_dimensions)):
        raise IOError('number of output files are not the same as the number of expected output xarrays')

    output_dims = { key : (output_dimensions[key]['coords'] ) for key in output_dimensions.keys()}
    output_dims_final = {}

    xarrays_output_dims = []
    for ixarray_out,xarray_output_dimensions in enumerate(xarrays_output_dimensions):
        xarrays_output_dims.append(
            {key: (xarray_output_dimensions[key]['coords'] ) for key in xarray_output_dimensions.keys()}
        )

    def get_fake_xarrays_out(xarrays_output_dims):
        xarrays_out = []
        for ixarray_out in range(len(xarrays_output_dims)):
            shape = []
            dims = []
            for dim,coord in xarrays_output_dims[ixarray_out].items():
                if coord is not None:
                    shape.append(len(coord))
                    dims.append(dim)
            size = np.product(shape)
            nbytes = 4 * size
            xarrays_out.append(
                Namespace(**{
                               'coords':  xarrays_output_dims[ixarray_out],
                               'shape':shape,
                               'dims':dims,
                               'size': size,
                                'nbytes':nbytes}
                          )
            )
        return xarrays_out
    logging.debug('Creating fake xarray outputs, '
                 'which we use to determine the no apply chunk sizes and the expected memory usage.')
    xarrays_out = get_fake_xarrays_out(xarrays_output_dims)


    def get_xarrays_shapes(
            xarrays,
            dims_apply_names,
            dims_no_apply,
            output_dimensions,
            number_of_chunks_apply_dims

    ):
        xarrays_shapes_chunks = [list() for i in range(len(xarrays))]
        xarrays_shapes = [list() for i in range(len(xarrays))]
        # xarrays_in_shapes_chunks_cumulative = [1]*len(xarrays)
        # xarrays_dims_transposed = [[]]*len(xarrays)

        xarrays_chunks_apply = [False] * len(xarrays)
        for ixarray, xarray in enumerate(xarrays):
            for idim, dimname in reversed(list(enumerate(dims_apply_names))):
                # print(dimname,ixarray,xarrays_in_shapes_chunks)
                if dimname in xarray.dims:
                    if (

                            (dimname in output_dimensions) and
                            # (output_dimensions[dimname] != None) and
                            ('chunksize' in list(output_dimensions[dimname].keys())) and
                            (dimname in number_of_chunks_apply_dims.keys()) and \
                            identical_xarrays(xarray.coords[dimname], output_dimensions[dimname]['coords'])
                    ):
                        xarrays_shapes_chunks[ixarray].insert(0, output_dimensions[dimname]['chunksize'])
                        # if 'overlap' in output_dimensions[dimname]:
                        #     xarrays_shapes_chunks[ixarray][0] += output_dimensions[dimname]['overlap']

                        xarrays_chunks_apply[ixarray] = True
                    else:
                        # if dimname in xarray.dims: # this case already caught above
                        xarrays_shapes_chunks[ixarray].insert(0, len(xarray.coords[dimname]))
                        xarrays_shapes[ixarray].insert(0, xarray.shape[xarray.dims.index(dimname)])
                    # else:
                    #     xarrays_shapes_chunks[ixarray].insert(0, None)
                    #     xarrays_shapes[ixarray].insert(0,None)#xarray.shape[xarray.dims.index(dimname)])
                else:
                    xarrays_shapes_chunks[ixarray].insert(0, None)
                    xarrays_shapes[ixarray].insert(0, None)

        if number_of_chunks_apply > 1:
            for ixarray, xarray in enumerate(xarrays):
                if xarrays_chunks_apply[ixarray] == True:
                    xarrays_shapes_chunks[ixarray].insert(0, number_of_chunks_apply)
                else:
                    xarrays_shapes_chunks[ixarray].insert(0, None)
                xarrays_shapes[ixarray].insert(0, None)

        for idim, dimname in reversed(list(enumerate(dims_no_apply))):
            for ixarray, xarray in enumerate(xarrays):
                if dimname != '__chunk__':  # inner extra chunk dimension is already considered in previous loop
                    if dimname in xarray.dims:
                        xarrays_shapes_chunks[ixarray].insert(0, len(xarray.coords[dimname]))
                        xarrays_shapes[ixarray].insert(0, len(xarray.coords[dimname]))
                    else:
                        xarrays_shapes_chunks[ixarray].insert(0, None)
                        xarrays_shapes[ixarray].insert(0, None)
        return xarrays_shapes, xarrays_shapes_chunks

    xarrays_in_shapes, xarrays_in_shapes_chunks = get_xarrays_shapes(
        xarrays_in, dims_apply_names, dims_no_apply, output_dimensions,number_of_chunks_apply_dims)
    xarrays_out_shapes, xarrays_out_shapes_chunks = get_xarrays_shapes(
        xarrays_out, dims_apply_names, dims_no_apply, output_dimensions,number_of_chunks_apply_dims)

    logging.debug('xarrays shapes for '+str(dims_no_apply.keys()) +' + '+str(dims_apply_names)+' : ')
    logging.debug('  -> original xarrays in: '+str(xarrays_in_shapes))
    logging.debug('  ->  chunked xarrays in: '+str(xarrays_in_shapes_chunks))
    logging.debug('  -> original xarrays out: '+str(xarrays_out_shapes))
    logging.debug('  ->  chunked xarrays in: '+str(xarrays_out_shapes_chunks))
    logging.debug('determining input chunk format that fits our maximum memory size input of '+str(maximum_memory_size_per_proc_bytes))

    xarrays_all = list(xarrays_in)+list(xarrays_out)
    xarrays_shapes_chunks_all = xarrays_in_shapes_chunks+xarrays_out_shapes_chunks
    xarrays_shapes = xarrays_in_shapes+xarrays_out_shapes

    # # WORKAROUND to avoid memory cacheing big files when using 'nbytes'
    # chunks_memory_sizes =  [int(xarray.nbytes/max(xarray.size,1)) for xarray in xarrays_all]
    # chunks_memory_sizes_dim = [[int(xarray.nbytes/max(xarray.size,1))] for xarray in xarrays_all]
    chunks_memory_sizes = []
    chunks_memory_sizes_dim = []
    for xarray in xarrays_all:
        if type(xarray) == Namespace:
            nbytes = xarray.nbytes
        elif xarray.dtype == 'float32':
            nbytes = 4
        elif xarray.dtype == 'float64':
            nbytes = 8
        else:
            logging.warning('dtype cannot be tracked from xarray. retrieving xarray.nbytes could lead to (possibly only when using dataarray_wrappers)')
            nbytes = xarray.nbytes
        chunks_memory_sizes.append(int(nbytes/max(xarray.size,1)))
        chunks_memory_sizes_dim.append([int(nbytes/max(xarray.size,1))])

    iteration_over_apply_dims = list(reversed(list(enumerate(dims_all))[-len(dims_apply_names):]))
    for idim,dimname in iteration_over_apply_dims:
        for ixarray,xarray in enumerate(xarrays_all):
            if xarrays_shapes_chunks_all[ixarray][idim] != None:
                chunks_memory_sizes[ixarray] *= xarrays_shapes_chunks_all[ixarray][idim]
                chunks_memory_sizes_dim[ixarray].insert(0,xarrays_shapes_chunks_all[ixarray][idim])

    iteration_over_noapply_dims = list(reversed(list(enumerate(dims_all))[:len(dims_no_apply)]))
    current_memory_size  = sum(chunks_memory_sizes)

    chunk_sizes_no_apply = {}
    for idim,dimname in iteration_over_noapply_dims:
        if (current_memory_size < maximum_memory_size_per_proc_bytes):
            chunks_memory_sizes_total = sum(chunks_memory_sizes)
            xarrays_sized_cumulative_base = 0
            xarrays_sized_cumulative_mul = 0
            for ixarray,xarray in enumerate(xarrays_all):
                #xarrays_sizes_cumulative[ixarray] *= xarrays_in_shapes_chunks[ixarray][idim]
                if xarrays_shapes_chunks_all[ixarray][idim] == None:
                    xarrays_sized_cumulative_base += chunks_memory_sizes[ixarray]
                else:
                    xarrays_sized_cumulative_mul += chunks_memory_sizes[ixarray]

            if dimname == '__chunk__':
                logging.debug('We do not allow grouping of apply-dimension chunking.')
                chunk_sizes_no_apply[dimname] = 1.0
            else:
                if xarrays_sized_cumulative_mul  != 0:
                    chunk_sizes_no_apply[dimname] = np.floor(np.min([(maximum_memory_size_per_proc_bytes - xarrays_sized_cumulative_base)/xarrays_sized_cumulative_mul, dims_no_apply_lengths[dimname]]))
                else:
                    chunk_sizes_no_apply[dimname] = 0

            if not chunk_sizes_no_apply[dimname].is_integer():
                raise ValueError('whole number expected for dimension selection size')

            current_memory_size = int((xarrays_sized_cumulative_base + xarrays_sized_cumulative_mul * chunk_sizes_no_apply[dimname]))
            #print(current_memory_size > maximum_memory_size)
            # if current_memory_size > maximum_memory_size:
            #     raise ValueError('something wrong with the chunk size calculation for limiting memory')
            for ixarray,xarray in enumerate(xarrays_all):
                if xarrays_shapes_chunks_all[ixarray][idim] != None:
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

    logging.debug('overall memory size: '+ str(current_memory_size))
    logging.debug('xarray chunk memory sizes for arrays: '+ str(chunks_memory_sizes))
    logging.debug('xarray chunk memory size per dimension (last one is character byte size): '+ str(chunks_memory_sizes_dim))

    chunks_number_no_apply = {}
    for dimname,shape in dims_no_apply_lengths.items():
        chunks_number_no_apply[dimname] = np.ceil(dims_no_apply_lengths[dimname]/chunk_sizes_no_apply[dimname])

    logging.debug('memory input size of chunks: '+ str(current_memory_size) +'/'+ \
                 str(maximum_memory_size_per_proc_bytes) +' = '+str(current_memory_size/int(maximum_memory_size_per_proc_bytes)*100)+'% of maximum \n'+ \
                 ' - expected memory usage: ' + str(current_memory_size) + '\n' + \
                 ' - limit of memory usage: ' + str(current_memory_size) + '\n' + \
                 ' - chunks_memory_sizes: ' + str(chunks_memory_sizes) + '\n' + \
                 ' - chunks_memory_sizes_dim' + str(chunks_memory_sizes) + '\n' + \
                 'Please consider the usage of memory chunking along the apply_dimensions'
                 )
    if current_memory_size > maximum_memory_size_per_proc_bytes:
        logging.warning('expected memory usage exceeds predefined memory limit!')
        if not ignore_memory_limit:
            raise IOError('memory limit needs to be respected. Or turn on ignore_memory_limit')

    chunks_no_apply = list(product(*tuple([list(range(int(a))) for a in list(chunks_number_no_apply.values())])))

    first_chunks = True
    # logging.info('closing xarrays_out, which we used above to easily calculate the dimension/memory shaping. Data is written out directly through disk for saving memory. ')
    # for ixarray_out, xarray_out in enumerate(xarrays_out):

    #     xarrays_out[ixarray_out].close()

    # logging.info('Re-opening netcdf files for output
    # for ixarray_out, xarray_out in enumerate(xarrays_out):
    #     ncouts[ixarray_out] = nc4.Dataset(xarrays_output_filenames_work[ixarray_out],'a')




    def divide_in_groups(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    logging.debug('initialize lists for output netcdfs')
    ncouts = []
    ncouts_variable = []
    xarrays_output_filenames_work = []
    xarrays_output_filenames_real = []
    xarrays_output_dims_final = []



    chunks_no_apply_groups = list(divide_in_groups(chunks_no_apply,nprocs))

    for index_no_apply_group in tqdm.tqdm(chunks_no_apply_groups,position=barposition):
       if nprocs > 1:
           pool = Pool()
           iterate_func = pool.map(partial(chunk_task, func,
                      chunks_number_no_apply, dims_apply_names, number_of_chunks_apply_dims, output_dimensions,
                          dims_no_apply_lengths, chunk_sizes_no_apply, xarrays_in, xarrays_in_shapes_chunks, dims_all,dims_no_apply,pass_missing_output_coordinates,
                                                                                 args_func,kwargs_func), tuple(index_no_apply_group))
       else:
           iterate_func = [chunk_task(func,
                      chunks_number_no_apply, dims_apply_names, number_of_chunks_apply_dims, output_dimensions,
                          dims_no_apply_lengths, chunk_sizes_no_apply, xarrays_in, xarrays_in_shapes_chunks, dims_all,dims_no_apply,pass_missing_output_coordinates,args_func,kwargs_func,index_no_apply_group[0],)]


       for (chunks_out_xarrays, chunk_start, chunk_end)  in iterate_func:

           if (xarrays_output_filenames is not None) and (xarrays_output_filenames != False) and (type(xarrays_output_filenames) != str) and (
                   len(chunks_out_xarrays) != len(xarrays_output_filenames)):
               raise ValueError('The number of outputs from function (' + str(
                   len(chunks_out_xarrays)) + ') is different from the number' + \
                                ' of output filenames (' + str(len(xarrays_output_filenames)) + ')')
           if first_chunks == True:
               logging.debug('recreate xarrays_output_dims definition according to actual function output')
               xarrays_output_dims_final = []
               for ichunk_out,chunk_out_xarray in enumerate(chunks_out_xarrays):
                   xarrays_output_dims_final.append({})
                   for dimname in output_dims.keys():
                       if dimname in chunk_out_xarray.dims:
                           # (output_dimensions[dimname] != None) and
                           if (dimname in dims_apply_names):
                               if (dimname in output_dimensions) and \
                                       ('chunksize' in list(output_dimensions[dimname].keys())):
                                   if not identical_xarrays(
                                       chunk_out_xarray.coords[dimname],
                                       output_dimensions[dimname]['coords'][chunk_start[dims_all.index(dimname)]:chunk_end[dims_all.index(dimname)]]
                                       ):
                                       raise ValueError('output dimensions are not identical to input. We may get into trouble!!!')
                                   xarrays_output_dims_final[ichunk_out][dimname] = xarrays_output_dims[ichunk_out][dimname]
                               else:
                                   xarrays_output_dims_final[ichunk_out][dimname] = chunk_out_xarray.coords[dimname]
                           elif (dimname in dims_no_apply.keys()):
                               logging.debug('assuming that chunks over dimension '+str(dimname)+' should be concatenated in the final xarrray output')
                               xarrays_output_dims_final[ichunk_out][dimname] = xarrays_output_dims[ichunk_out][dimname]
                           else:
                               raise ValueError('Unexpected chunk dimensions')
                               #xarrays_output_dims_final[ichunk_out][dimname] = chunk_out_xarray.coords[dimname]
                               #dimname in dims_no_apply.keys():
                       else:
                           xarrays_output_dims_final[ichunk_out][dimname] = None
                       output_dims_final[dimname] = output_dims[dimname]

                   for dimname in chunk_out_xarray.dims:
                       if dimname not in output_dims.keys():
                           output_dims_final[dimname] = chunk_out_xarray.coords[dimname]

                       if dimname not in xarrays_output_dims[ichunk_out].keys():
                          xarrays_output_dims_final[ichunk_out][dimname] = chunk_out_xarray.coords[dimname]


                   for dimname in chunk_out_xarray.dims:
                       if dimname not in xarrays_output_dims_final[ichunk_out].keys():
                           raise IOError ( 'unexpected dimension in function output: '+dimname+'.')
                   for dimname,coord in xarrays_output_dims[ichunk_out].items():
                       if (xarrays_output_dims_final[ichunk_out][dimname] is None):
                           if (coord is not None):
                               logging.warning('dimension '+dimname+' is not detected in the function output, while this was expected. Memory usage '+\
                                           ' may not have been optimal. For optimal usage, specify the output coordinates '+\
                                           '(xarrays_)output_dimensions = {"' + dimname + '" : { "coords" : None }')

                       elif not identical_xarrays(coord,xarrays_output_dims_final[ichunk_out][dimname]):

                           logging.warning('xarrays_output_dims ----' )
                           logging.warning(str(coord))
                           logging.warning('end xarrays_output_dims ----' )
                           logging.warning('xarrays_output_dims_final ----' )
                           logging.warning(str(xarrays_output_dims_final[ichunk_out][dimname]))
                           logging.warning('end xarrays_output_dims_final ---- dimension: '+dimname+' of xarray output number '+str(ichunk_out))
                           logging.warning('dimension of function output (xarrays_output_dims_final) is different from ')
                           logging.warning('what is expected (xarrays_output_dims). Memory usage '+\
                                           'may not have been optimal. For optimal usage, specify the output coordinates '+\
                                           '(xarrays_)output_dimensions = {"'+dimname+'" : {"coords": xarrays_output_dims_final[ichunk_out][dimname]}}')
               xarrays_out_final = get_fake_xarrays_out(xarrays_output_dims_final)



           xarrays_out_final_shapes,xarrays_out_final_shapes_chunks = get_xarrays_shapes(
               xarrays_out_final,dims_apply_names,dims_no_apply,output_dimensions,number_of_chunks_apply_dims,
           )

           xarrays_out_selection_chunk = get_xarrays_selection_chunk(
               xarrays_out_final,
               xarrays_out_final_shapes_chunks,
               dims_all,
               output_dimensions,
               dims_apply_names,
               number_of_chunks_apply_dims,
               dims_no_apply,
               chunk_start,
               chunk_end,
           )

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


           # that was duplicate code???
           # if type(chunks_out).__name__ not in ['tuple', 'list']:
           #     # list_output = False
           #     chunks_out= [chunks_out]


           for ichunk_out,chunk_out_xarray in enumerate(chunks_out_xarrays):

               # for idim, dimname in reversed(list(enumerate(dims_no_apply))):
               #     for ixarray, xarray in enumerate(xarrays_all):
               #         if dimname != '__chunk__':  # inner extra chunk dimension is already considered in previous loop
               #             if dimname in xarray.dims:
               #                 xarrays_in_shapes_chunks[ixarray].insert(0, len(xarray.coords[dimname]))
               #                 xarrays_shapes[ixarray].insert(0, len(xarray.coords[dimname]))
               #             else:
               #                 xarrays_in_shapes_chunks[ixarray].insert(0, None)
               #                 xarrays_shapes[ixarray].insert(0, None)

               logging.debug('xarray selection of chunk output '+str(ichunk_out)+': ' + str(xarrays_out_selection_chunk[ichunk_out]))

               xarrays_out_selection_chunk_ordered = sort_dict_by_keys(xarrays_out_selection_chunk[ichunk_out],list(xarrays_output_dimensions[ichunk_out].keys()))
               #if type(chunk_out) == type(np.array([])):
               # if type(chunk_out) != xr.core.dataarray.DataArray:
               #     chunk_out_coordinates = {}
               #     for dimname in xarrays_out_selection_chunk[ichunk_out].keys():
               #         chunk_out_coordinates[dimname] = xarrays_output_dims_final[ichunk_out][dimname].isel({dimname:xarrays_out_selection_chunk[ichunk_out][dimname]})

               #     chunk_out_xarray = xr.DataArray(chunk_out,coords=chunk_out_coordinates)
               # else:
               #     chunk_out_xarray = chunk_out

               logging.debug('xarray selection ordered for output array '+str(ichunk_out)+': ' + str(xarrays_out_selection_chunk_ordered))


           for ichunk_out,chunk_out_xarray in enumerate(chunks_out_xarrays):
               chunk_out_xarray_ordered = chunk_out_xarray.transpose(*tuple(xarrays_out_selection_chunk_ordered.keys()))
               # chunk_profile =

               logging.debug('re-ordered output shape: '+str(chunk_out_xarray.shape) +' -> '+ str(chunk_out_xarray_ordered.shape))
               indexing_for_output_array = tuple([dim_selection for dim_selection in xarrays_out_selection_chunk_ordered.values()])
               logging.debug('index of chunk in netcdf output '+str(ichunk_out)+': ' + str(indexing_for_output_array))
               #logging.debug('this should fit in netcdf total output shape '+str(ncouts[ichunk_out].variables['__xarray_data_variable__'].shape))
               logging.debug('this should fit in netcdf total output shape '+str(xarrays_out[ichunk_out].shape))

               overlap_weights = np.ones_like(chunk_out_xarray_ordered.values,dtype=float)
               idim = 0
               for dim,selection_chunk_out in xarrays_out_selection_chunk_ordered.items():
                   # overlap_weights_dim = np.ones((len(xarrays_out_selection_chunk_ordered[dim],)))
                   #reshape(list(range(idim-1))+overlap_weights.shape[idim])
                   if ('overlap' in output_dimensions[dim]) and (dim in number_of_chunks_apply_dims.keys()):

                       if profile_overlap == 'triangle':
                           if xarrays_out_selection_chunk_ordered[dim][0] == 0:
                               left = np.ones(output_dimensions[dim]['overlap'])
                           else:
                               left = np.arange(0.,output_dimensions[dim]['overlap'],1.)/output_dimensions[dim]['overlap']

                           middle = np.ones((max(0,(len(xarrays_out_selection_chunk_ordered[dim]) - 2 * output_dimensions[dim]['overlap']),)))

                           if xarrays_out_selection_chunk_ordered[dim][-1] == (len(output_dimensions[dim]['coords']) - 1):

                               #at the right border of the dimension, we do just the remaning, hence -len(left)-len(middle)
                               right = np.ones(min(output_dimensions[dim]['overlap'] , max(0,len(xarrays_out_selection_chunk_ordered[dim])-len(left)-len(middle))))
                           else:
                               right = np.arange(min(output_dimensions[dim]['overlap'] , max(0,len(xarrays_out_selection_chunk_ordered[dim])-len(left)-len(middle))),0,-1)/output_dimensions[dim]['overlap']
                       elif profile_overlap == 'square':
                           if xarrays_out_selection_chunk_ordered[dim][0] == 0:
                               left = np.ones(output_dimensions[dim]['overlap'])
                           else:
                               leftleft = int(output_dimensions[dim]['overlap']/2)
                               leftright = output_dimensions[dim]['overlap'] - leftleft
                               left = np.concatenate([np.zeros(leftleft),np.ones(leftright)])

                           middle = np.ones(output_dimensions[dim]['chunksize'] - 2 * output_dimensions[dim]['overlap'])

                           # right = np.arange(min(output_dimensions[dim]['overlap'] , max(0,len(xarrays_out_selection_chunk_ordered[dim])-len(left)-len(middle))),0,-1)/output_dimensions[dim]['overlap']
                           leftright = int(output_dimensions[dim]['overlap']/2)
                           rightright = output_dimensions[dim]['overlap'] - leftright
                           right = np.concatenate([np.ones(leftright),np.zeros(rightright)])
                       else:
                           raise ValueError ('Profile overlap '+ str(profile)+ 'not implemented')

                       overlap_weights_dim = np.concatenate([left,middle,right])[:len(xarrays_out_selection_chunk_ordered[dim])]
                       overlap_weights_dim = overlap_weights_dim.reshape([1]*idim+[overlap_weights.shape[idim]]+[1]*(len(overlap_weights.shape) - idim -1))
                       try:
                        overlap_weights *= overlap_weights_dim
                       except:
                           import pdb; pdb.set_trace()
                   idim += 1

               if first_chunks == True:
                   attributes_out = {}

                   logging.debug('propagate attributes from xarray chunk function output')
                   attributes_out['variable']  = chunk_out_xarray_ordered.name
                   for attrkey,attrvalue in chunk_out_xarray_ordered.attrs.items():
                       attributes_out[attrkey] = attrvalue

                   logging.debug('update attributes derived from possible new coordinate system')
                   # xarray_out = xr.open_dataarray(xarrays_output_filenames_work)
                   # dimensions_attributes = get_dimensions_attributes(xarrays_output_dims[incout])
                   dimensions_attributes = get_dimensions_attributes(
                           xarrays_output_dims_final[ichunk_out],
                           time_id = (attributes_out['time_id'] if 'time_id' in attributes_out.keys() else None),
                           space_id = (attributes_out['space_id'] if 'space_id' in attributes_out.keys() else None)
                           )

                   # #  ???????????
                   # for dim in dims_apply_names:
                   #     # if dimensions_attributes[dim] is None:
                   #     #     import pdb; pdb.set_trace()
                   #     #     if dim not in attributes_out.keys():
                   #     #         dimensions_attributes[dim] = 'None'

                   #     #??????
                   #         if ((dim in xarrays_out[ichunk_out].dims) and not identical_xarrays(xarrays_out[ichunk_out].coords[dim],xarrays_output_dims_final[ichunk_out][dim])) and \
                   #                 (dim in dimensions_attributes.keys()):
                   #             attributes_out[dim] = dimensions_attributes[dim]


                   # if ('latitude' in dims_apply_names) or ('longitude' in dims_apply_names):
                   #     if (( 'latitude' in xarrays_in[0].dims) and (not identical_xarrays(xarrays_output_dims_final[ichunk_out]['latitude'],xarrays_in[0][dim]))) and \
                   #        (( 'longitude' in xarrays_in[0].dims) and ( not identical_xarrays(xarrays_output_dims_final[ichunk_out]['longitude'], xarrays_in[0][dim]))) and \
                   #        ('space' in dimensions_attributes.keys()):
                   #         attributes_out['space'] = dimensions_attributes['space']
                   # #  ???????????

                   logging.debug('adding attributes through apply_func input argument "attributes"')
                   if attributes != None:
                       logging.debug('assigning extra attributes...')
                       for attrkey, attrvalue in attributes[ichunk_out].items():
                           if type(attrvalue) == type(lambda x: x):
                               values_input = []
                               for xarray_in in xarrays_in:
                                   if key in xarray_in.attrs.keys():
                                       values_input.append(xarray_in.attrs[key])
                                   else:
                                       values_input.append(None)
                               attr_value_out = attrvalue(values_input)
                           else:
                               attr_value_out= attrvalue
                           if (attrkey in attributes_out.keys()) and (attributes_out[attrkey] is not None):
                               logging.warning('Attribute '+attrkey+' (value = '+str(attributes_out[attrkey])+' ) is already assigned manually in the attributes output of ichunk_out ' + str(ichunk_out) +'. Ignoring (value = "'+ str(attr_value_out)+'"). If you intend to have the latter, you need to remove the attribute manually from the output xarrays of your operator.')
                           else:
                            attributes_out[attrkey] = attr_value_out
                            logging.debug('ichunk_out ' + str(ichunk_out) + ' - ' + attrkey + ' - ' + str(attrvalue) + ' - ' +
                                         str(attributes_out[attrkey]))

                       logging.debug('...end assigning extra attributes')

                   for key in dimensions_attributes.keys():
                       if ((key not in attributes_out) or (attributes_out[key] == None)):
                               attributes_out[key] = dimensions_attributes[key]

                   logging.debug('building output for chunk number '+str(ichunk_out) )
                   xrtemp = xr.Dataset()
                   #for ichunk_out in range(len(xarrays_output_dimensions)):
                   for dimname, coords in xarrays_output_dims_final[ichunk_out].items():
                       if coords is not None:
                           xrtemp[dimname] = coords

                   for coordinates_key, coordinates in chunk_out_xarray_ordered.coords.items():
                       if coordinates_key not in xrtemp.dims:
                           xrtemp.coords[coordinates_key] = \
                               xr.DataArray(np.zeros( [len(xarrays_output_dims_final[ichunk_out][dim]) for dim in coordinates.dims]),dims=coordinates.dims)

                   ncout_dims = list()
                   ncout_shapes = list()
                   for key,value in xarrays_output_dims_final[ichunk_out].items():
                       if value is not None:
                           ncout_shapes.append(len(xarrays_output_dims_final[ichunk_out][key]))
                           ncout_dims.append(key)

                   # #ncout_coords = {}
                   # for coordname,coord in xrtemp.coords.items():
                   #     if coordname not in xrtemp.dims:
                   #         import pdb; pdb.set_trace()
                   #         xrtemp.coords[coordname] = {key:xarrays_output_dims_final[ichunk_out][key] for key in coord.dims}
                   #         #ncout_coords[key] = {key:xarrays_output_coords_final[ichunk_out][key] for key in coords.dims}
                   #         #xrtemp.coords[key] = ncout_coords[key]
                   
                   # for coordname,coord in xrtemp.coords.itme
                   #      ncout_coords

                   

                   logging.debug('acquiring variable attributes: '+str(attributes_out))

                   def fix_dict_for_ncattributes(attributes):
                       attributes_out = {}
                       for attrkey,attrvalue in attributes.items():
                           if ((type(attrvalue) == str) and ( attrvalue == '')) :
                               logging.warning('Excluding attribute "'+attrkey+'" that has empty value. Apparently, this gives problems when writing to the netcdf later on.')
                           elif (type(attrvalue) is not str):
                               logging.warning(
                                   'Excluding attribute "' + attrkey + '" that is not a string.')
                           else:
                               attributes_out[attrkey] = attrvalue
                       return attributes_out


                   attributes_out = fix_dict_for_ncattributes(attributes_out)

                   if 'variable' in attributes_out.keys():
                       chunk_out_xarray_ordered.name = attributes_out['variable']
                   else:

                       chunk_out_xarray_ordered.name = 'None'

                   if (xarrays_output_filenames is not None) and (xarrays_output_filenames != False):
                       logging.debug('Acquiring real output filename for xarray out number '+str(ichunk_out)+' and setting output (temporary filename)')
                       if type(xarrays_output_filenames) == str:
                           xarrays_output_filenames_pattern = xarrays_output_filenames
                       elif type(xarrays_output_filenames) in (list,tuple):
                           xarrays_output_filenames_pattern = xarrays_output_filenames[ichunk_out]

                       logging.debug('parsing from pattern: '+xarrays_output_filenames_pattern)
                       xarrays_output_filenames_real.append( name_from_pattern( xarrays_output_filenames_pattern,attributes_out))
                       logging.debug('filename output: '+xarrays_output_filenames_real[-1])

                       if os.path.isfile(xarrays_output_filenames_real[ichunk_out]):
                           if overwrite_output_filenames == False:
                               raise FileExistsError(
                                   xarrays_output_filenames_real[ichunk_out] + ' ( ' + xarrays_output_filenames[ichunk_out] + ' ) exists.'
                               )
                           else:
                              logging.warning(
                                  'Filename output ' + xarrays_output_filenames_real[ichunk_out] + ' (' + \
                                  xarrays_output_filenames[ ichunk_out] + ') exists. Removing before writing.'
                              )
                              os.system('rm '+xarrays_output_filenames_real[ichunk_out])
                   else: 
                       xarrays_output_filenames_real.append(False)

                   if xarrays_output_filenames_real[ichunk_out] == False:
                        xarrays_output_filenames_work.append(xarrays_output_filenames_real[ichunk_out])
                        logging.debug("Dumping output of chunk "+str(ichunk_out)+" directly into memory." )

                   if not tempfile_dir:
                       xarrays_output_filenames_work.append(xarrays_output_filenames_real[ichunk_out])
                       logging.debug("Dumping output directly to final destination: " + str(xarrays_output_filenames_work[-1]))
                   else:
                       logging.debug("Using temporary output dir, eg., good for working with network file systems")
                       if (tempfile_dir is None) or (tempfile_dir is True):
                           xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=None))
                           logging.debug("Using temporary output in default tempfile_dir: " + xarrays_output_filenames_work[-1])
                       else:
                           xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=tempfile_dir))
                           logging.debug("Using temporary output in specified tempfile_dir: " + xarrays_output_filenames_work[-1])

                   
                   if chunk_out_xarray_ordered.name is None:
                       raise ValueError('output name of chunk is None. please specify your_output_variable.name  .')
                   ncouts_variable.append(chunk_out_xarray_ordered.name)

                   if xarrays_output_filenames_work[ichunk_out] != False:
                       fnout = xarrays_output_filenames_work[ichunk_out]  # 'testing_'+str(iarray)+'.nc'
                       if os.path.isfile(fnout):
                           raise FileExistsError('output file ' + fnout + ' exists. Aborting... ')
                       # os.system('rm ' + fnout)
                       xrtemp.to_netcdf(fnout)
                       logging.info('creating netcdf file '+fnout)
                       ncouts.append(nc4.Dataset(fnout, 'a'))
                       ncouts[ichunk_out].createVariable(chunk_out_xarray_ordered.name, "f", tuple(ncout_dims),fill_value=0.)
                       for attrkey,attrvalue in attributes_out.items():
                           logging.info('writing netcdf attribute '+attrkey+' = '+str(attrvalue))
                           ncouts[ichunk_out].variables[ncouts_variable[-1]].setncattr(attrkey,attrvalue)
                       ncouts[ichunk_out].close()

                       #xarrays_out.append(xr.open_dataarray(fnout))
                       logging.info('finished initializing netcdf file '+str(ichunk_out))
                   else:
                       logging.info('We are not writing to disc but we store output directly in memory')
                       # ncouts.append(xrtemp)
                       # xrtemp[chunk_out_xarray_ordered.name] =
                       xrtemp[ncouts_variable[-1]] = xr.DataArray(np.zeros(ncout_shapes),name=ncouts_variable[-1],dims=tuple(ncout_dims))
                       ncouts.append(xr.zeros_like(xrtemp[ncouts_variable[-1]]))


                   xrtemp.close()
               # try:

               if type(ncouts[ichunk_out]) == nc4.Dataset:
                   logging.debug('acquiring previous values for consolidating chunk overlapping values')
                   ncouts[ichunk_out] = nc4.Dataset(xarrays_output_filenames_work[ichunk_out],'a')
                   recap = ncouts[ichunk_out].variables[ncouts_variable[ichunk_out]][indexing_for_output_array].filled(fill_value=0)

                   logging.debug('writing chunk ('+str(indexing_for_output_array)+') to netcdf file '+str(ichunk_out))
                   if first_chunks:
                       logging.info('writing first chunk ('+str(indexing_for_output_array)+') to netcdf file '+str(ichunk_out)+'. This takes a much longer than the next chunks because of some hidden initializations of the netcdf file.')
                   ncouts[ichunk_out].variables[ncouts_variable[ichunk_out]][indexing_for_output_array] = \
                       recap + np.array( chunk_out_xarray_ordered.values, dtype='float32') * overlap_weights

                   for coordname,coord in chunk_out_xarray_ordered.coords.items():
                       if coordname not in  chunk_out_xarray_ordered.dims:
                           indexing_for_output_array_coord = [indexing_for_output_array[chunk_out_xarray_ordered.dims.index(dim)] for dim in coord.dims]
                           if len(indexing_for_output_array_coord) > 0:
                               ncouts[ichunk_out].variables[coordname][indexing_for_output_array_coord] = coord.values
                       

                   if first_chunks:
                       logging.info('finished writing first chunk')
                   ncouts[ichunk_out].close()
               elif type(ncouts[ichunk_out]) is xr.DataArray:
                   recap = ncouts[ichunk_out][indexing_for_output_array]
                   ncouts[ichunk_out][indexing_for_output_array] = recap + np.array( chunk_out_xarray_ordered.values, dtype='float32') * overlap_weights
                   # for coordname,coord in chunk_out_xarray_ordered.coords.items():

                   # indexing_for_output_array
                   # xarrays_out_selection_chunk_ordered
                   # for coordinates_key, coordinates in chunk_out_xarray_ordered.coords.items():
                   #     ncouts[ichunk_out]

                   #     if coordinates_key not in xrtemp.dims:
                   #         xrtemp.coords[coordinates_key] = \
                   #             xr.DataArray(np.zeros( [len(xarrays_output_dims_final[ichunk_out][dim]) for dim in coordinates.dims]),dims=coordinates.dims)


               else:
                  raise IOError('type of output chunk '+str(ichunk_out)+'('+type(ncouts[ichunk_out]).__name__+') not inplemented.')
               #   except:
               #       import pdb; pdb.set_trace()
               del recap

               #del chunk_out_xarray
               del chunk_out_xarray_ordered
               logging.debug('.... finished')

           # for ichunk_out in range(len(chunks_out)):
           #     if type(chunks_out[ichunk_out]) == xr.core.dataarray.DataArray:
           #         chunks_out[ichunk_out].close()
           first_chunks = False

       if nprocs > 1:
           pool.close()

    xrouts = []
    for incout in range(len(ncouts)):
        if type(ncouts[incout]) == nc4.Dataset:

            logging.warning('workaround with _FillValue to enable overlapping values')
            ncouts[incout] = nc4.Dataset(xarrays_output_filenames_work[incout],'a')
            ncouts[incout][ncouts_variable[incout]].delncattr('_FillValue')
            ncouts[incout].close()
            if not os.path.isdir(os.path.dirname(xarrays_output_filenames_real[incout])):
                CMD = 'mkdir -p '+os.path.dirname(xarrays_output_filenames_real[incout])
                logging.info('Creating destination folder: ' + CMD)
                os.system(CMD)

            CMD = 'mv '+xarrays_output_filenames_work[incout]+' '+xarrays_output_filenames_real[incout]
            logging.info('Moving temporary output to actual netcdf: '+CMD)
            os.system(CMD)
            sleep(10)
            xrouts.append(xr.open_dataarray(xarrays_output_filenames_real[incout]))
        elif type(ncouts[incout]) == xr.DataArray:
            xrouts.append(ncouts[incout])

    del xarrays_output_dimensions
    if return_type == 'xarrays':
        return tuple(xrouts)
    else:
        for xrout in xrouts:
            xrout.close()
            del xrout
        if return_type == 'paths':
            return xarrays_output_filenames_real
        else:
            raise ValueError('Return type '+return_type+' not supported.')
    #return xarrays_output_filenames_real


# if __name__ == '__main__':
#     input_file = '/home/woutersh/projects/KLIMPALA_SF/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
#     ds = xr.open_dataset( input_file,)['Band1'].rename({'lat':'latitude'}).rename({'lon':'longitude'})
#     ds = ds.isel(
#         latitude  = ( (ds.latitude  > -5) & (ds.latitude  < 5)),
#         longitude = ( (ds.longitude > -10) & (ds.longitude < 5))
#     )
#
#     #input_file = '/projects/C3S_EUBiodiversity/data/case_klimpala/aggregation-30-years/indicators-annual/cropped_to_africa/bias_corrected/cmip5_/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
#     input_file = '/home/woutersh/projects/KLIMPALA_SF/data/test/temperature--mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
#     ds2 = xr.open_dataarray(input_file)
#
#     # this also sets the order of (inner) output dimensions as expected by the function
#     dims_apply_names = ['latitude','longitude']
#
#     ignore_memory_limit = False
#
#     # this also defines the order in which the output dimensions are being constructed, allowing for transposing the output on the fly.
#     output_dimensions = { #input
#         #        'time':{ 'coords':ds2.time},
#         'longitude':{ 'coords':ds.longitude,'chunksize':1000,'overlap':50},
#         'latitude':{ 'coords':ds.latitude,'chunksize':500,'overlap':50},
#     }
#     #xarrays = [ds2,ds.latitude,ds.longitude]
#     xarrays = [ds2,ds.latitude,ds.longitude]
#
#     xarrays_output_dimensions = [] #default
#     #xarrays_output_filenames =
#
#     def func(x_coarse, latitude,longitude):
#         out = (pcd.vectorized_functions.extend_crop_interpolate( \
#             x_coarse.values, \
#             (x_coarse.latitude.values, x_coarse.longitude.values), \
#             (latitude.values, longitude.values), \
#             # interpolation=True,
#             # return_grid_output=False,
#             # debug=False,
#             # border_pixels=5,
#             # ascending_lat_lon = False,
#             # tolerance_for_grid_match = 1.e-9
#         ))
#         coords = dict(x_coarse.coords)
#         coords['latitude'] =  latitude
#         coords['longitude'] =  longitude
#         xrout = xr.DataArray(out,dims=x_coarse.dims,coords=coords)
#         xrout.name = x_coarse.name
#         xrout.attrs = x_coarse.attrs
#         return xrout, xrout
#     apply_func(
#         func,
#         xarrays,
#         dims_apply_names = ['latitude','longitude'],
#         xarrays_output_filenames = [
#             '/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing.nc',
#             '/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing2.nc'],
#         #attributes = None,
#         output_dimensions=output_dimensions,
#         maximum_memory_size_per_proc_bytes=2 * 10 ** 7,
#     #squeeze_apply_dims = False,
#         tempfile_dir='/tmp/',
#         overwrite_output_filenames=True,
#         pass_missing_output_coordinates=False,
#     )
#
