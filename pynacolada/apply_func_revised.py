#import dask.array as da
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
#input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
input_file = '/home/woutersh/projects/KLIMPALA_SF/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
ds = xr.open_dataset( input_file,)['Band1'].rename({'lat':'latitude'}).rename({'lon':'longitude'})

#input_file = '/projects/C3S_EUBiodiversity/data/case_klimpala/aggregation-30-years/indicators-annual/cropped_to_africa/bias_corrected/cmip5_daily/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
input_file = '/home/woutersh/projects/KLIMPALA_SF/data/test/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
ds2 = xr.open_dataarray(input_file)
xarrays = [ds,ds2]

dims_apply_names = ['latitude','longitude'] #input


coordinates_output = None
coordinates_output = { #input
#        'time':{ 'coordinates':ds2.time},
        'latitude':{ 'coordinates':ds.latitude,'chunksize':3500},
        'longitude':{ 'coordinates':ds.longitude,'chunksize':3500},
        }


xarrays_coordinates_output = None
# xarrays_coordinates_output = [ #input
#     {
#         'time': {'coordinates': ds2.time},
#         'latitude': {'coordinates': ds.latitude, 'chunksize': 4500},
#         'longitude': {'coordinates': ds.longitude, 'chunksize': 4500},
#     }
# ]

logging.debug('determining inner number of chunks')

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

logging.info('collecting the dimensions occuring in the xarrays over which the function is not applied')
dims_no_apply  = {}

for ixarray,xarray in enumerate(xarrays):
    for dimname in xarray.dims:
        if (dimname not in dims_no_apply) and (dimname not in dims_apply_names):
            dims_no_apply[dimname] = xarray[dimname]
        if (dimname in dims_no_apply) and (not identical_xarrays(xarray[dimname],dims_no_apply[dimname])):
            raise ValueError('dimension '+dimname+' of xarray number '+str(ixarray)+' is not the same as previously detected dimensions.')


# if coordinates_output is None:
#     coordinates_output_sort_keys = list(coordinates_output.keys())
# else:
#     coordinates_output_sort_keys = []

if xarrays_coordinates_output is not None:
    coordinates_output_new = {}
    for dimdict in xarrays_coordinates_output:
        for dimname in dimdict.keys():
            if dimname not in coordinates_output_new.keys():
                logging.info('adding missing coordinates_output from xarrays_coordinates_output for '+dimname+': '+dimdict[dimname])
                coordinates_output_new[dimname] = dimdict[dimname]
    logging.debug('overriding values and order of previous coordinates_output.')

    if coordinates_output is None:
        coordinates_output = {}

    for coordinates_output_orig in coordinates_output.keys():
        coordinates_output_new[coordinates_output_orig] = coordinates_output[coordinates_output_orig]
    coordinates_output = sort_dict_by_keys(coordinates_output_new,list(coordinates_output.keys()))


coordinates_output_new = {}
for dimname,coordinates in dims_no_apply.items():

    if dimname not in coordinates_output_new.keys():
        logging.info('adding no apply dimensions to the default coordinates_output '+dimname)
        coordinates_output_new[dimname] = {'coordinates':coordinates}

if coordinates_output is None:
    coordinates_output = {}

for coordinates_output_orig in coordinates_output.keys():
    coordinates_output_new[coordinates_output_orig] = coordinates_output[coordinates_output_orig]
coordinates_output = sort_dict_by_keys(coordinates_output_new, list(coordinates_output.keys()))

coordinates_output_new = {}
for dimname in dims_apply_names:
    for xarray in xarrays:
        if dimname in xarray.coords.keys():
            if dimname not in coordinates_output_new.keys():
                logging.info('adding apply dimensions from input xarray to the default coordinates_output ' + dimname)
                coordinates_output_new[dimname] = {'coordinates': xarray[dimname]}

if coordinates_output is None:
    coordinates_output = {}

for coordinates_output_orig in coordinates_output.keys():
    coordinates_output_new[coordinates_output_orig] = coordinates_output[coordinates_output_orig]
coordinates_output = sort_dict_by_keys(coordinates_output_new, list(coordinates_output.keys()))



    # for ixarray_output,xarray_coordinates_output in xarrays_coordinates_output:
    #     for dimname,coordinates in xarray_coordinates_output.items():
    #         if dimname not in coordinates_output.keys():
    #             coordinates_output[dimname] = xarray_coordinates_output[ixarray_output][dimname]

number_of_chunks_apply = 1
number_of_chunks_apply_dims = {}
for dimname,dimattr in coordinates_output.items():
    if 'chunksize' in coordinates_output[dimname]:
        number_of_chunks_apply_dims[dimname] = int(np.ceil(len(coordinates_output[dimname]['coordinates'])/coordinates_output[dimname]['chunksize']))
        number_of_chunks_apply *= number_of_chunks_apply_dims[dimname] #np.ceil(len(coordinates_output[dimname]['coordinates'])/coordinates_output[dimname]['chunksize'])

number_of_chunks_apply = int(number_of_chunks_apply)

logging.info('Adding extra chunking as a separate dimension')
if number_of_chunks_apply > 1:
    dims_no_apply['__chunk__'] = xr.DataArray(range(number_of_chunks_apply))

dims_all = list(dims_no_apply.keys()) + dims_apply_names
dims_no_apply_lengths = { name:dim.shape[0] for (name,dim) in dims_no_apply.items()}
# dims_apply_shapes = { name:dim.shape for (name,dim) in dims_apply_names.items()}
#dims_all_shapes = {**dims_no_apply_shapes,**dims_apply_shapes}


# for dimname,coordinates in dims_no_apply.items():
#     if dimname not in coordinates_output.keys():
#         logging.info('adding no apply dimensions to the default coordinates_output '+dimname+': '+coordinates)
#         coordinates_output[dimname] = {'coordinates':coordinates}

if xarrays_coordinates_output is None:
    logging.info('No coordinates output xarrays are set manually, so we guess them from the coordinates_output.'
                 'We do this here already so that we can take it into account in the memory size and optimal chunking.')
    xarrays_coordinates_output = [coordinates_output]

filenames_out = ['/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing.nc']


if len(filenames_out) != len(xarrays_coordinates_output):
    raise IOError('number of output files are not the same as the number of expected output xarrays')


filenames_out_work = []
directory_work = '/tmp/'

logging.info('Setting working output files')

xarrays_out = []

if filenames_out is not None:
    ncouts = []
    filenames_out_work = []
    for filename_out in filenames_out:
        if not directory_work:
            filenames_out_work.append(filename_out)
            logging.info("Dump output directly to final destination: " + filenames_out_work[-1])
        else:
            logging.info("Using temporary output dir, eg., good for working with network file systems")
            if (directory_work is None) or (directory_work is True):
                filenames_out_work.append(tempfile.mktemp(suffix='.nc', dir=None))
                logging.info("Using temporary output in default directory_work: " + filenames_out_work[-1])
            else:
                filenames_out_work.append(tempfile.mktemp(suffix='.nc', dir=directory_work))
                logging.info("Using temporary output in specified directory_work: " + filenames_out_work[-1])

    xrtemp = xr.Dataset()
    for ixarray_out in range(len(xarrays_coordinates_output)):
        for dimname, dim in xarrays_coordinates_output[ixarray_out].items():
            xrtemp[dimname] = dim['coordinates']
            # ncouts[iarray].createDimension(dim,shapes_out_transposed[iarray][idim])
            # ncouts[iarray].createVariable(dim,'d',(dim,),)
            # ncouts[iarray].variables[dim][:] = coords_out_transposed[iarray][idim]
        fnout = filenames_out_work[ixarray_out]  # 'testing_'+str(iarray)+'.nc'
        if os.path.isfile(fnout):
            raise IOError('output file ' + fnout + ' exists. Aborting... ')
        # os.system('rm ' + fnout)
        xrtemp.to_netcdf(fnout)
        xrtemp.close()
        ncouts.append(nc4.Dataset(fnout, 'a'))
        ncouts[ixarray_out].createVariable('__xarray_data_variable__', "f", tuple(xarrays_coordinates_output[ixarray_out].keys()))
        ncouts[ixarray_out].close()
        xarrays_out.append(xr.open_dataarray(fnout))


xarrays_shapes_in_chunks = [list() for i in range(len(xarrays+xarrays_out))]
xarrays_shapes = [list() for i in range(len(xarrays+xarrays_out))]
#xarrays_shapes_in_chunks_cumulative = [1]*len(xarrays)
#xarrays_dims_transposed = [[]]*len(xarrays)


xarrays_chunks_apply = [False]*len(xarrays+xarrays_out)
for ixarray,xarray in enumerate(xarrays+xarrays_out):
    for idim,dimname in reversed(list(enumerate(dims_apply_names))):
        #print(dimname,ixarray,xarrays_shapes_in_chunks)
        if dimname in xarray.dims:
            if ( 
                    (dimname in coordinates_output) and
                    ('chunksize' in list(coordinates_output[dimname].keys())) and
                    identical_xarrays(xarray[dimname],coordinates_output[dimname]['coordinates'])
               ):
                xarrays_shapes_in_chunks[ixarray].insert(0,coordinates_output[dimname]['chunksize'])
                xarrays_chunks_apply[ixarray] = True
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray[dimname]))
            xarrays_shapes[ixarray].insert(0,xarray.shape[xarray.dims.index(dimname)])
        else:
            xarrays_shapes_in_chunks[ixarray].insert(0,None)
            xarrays_shapes[ixarray].insert(0,None)

import pdb; pdb.set_trace()

logging.info('adding size of __chunk__ dimensions')

if number_of_chunks_apply > 1:
    for ixarray,xarray in enumerate(xarrays+xarrays_out):
        if xarrays_chunks_apply[ixarray] == True:
            xarrays_shapes_in_chunks[ixarray].insert(0,number_of_chunks_apply)
        else:
            xarrays_shapes_in_chunks[ixarray].insert(0,None)
        xarrays_shapes[ixarray].insert(0,None)

for idim,dimname in reversed(list(enumerate(dims_no_apply))):
    for ixarray,xarray in enumerate(xarrays+xarrays_out):
        if dimname != '__chunk__': #inner extra chunk dimension is already considered in previous loop
            if dimname in xarray.dims:
                xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray[dimname]))
                xarrays_shapes[ixarray].insert(0, len(xarray[dimname]))
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,None)
                xarrays_shapes[ixarray].insert(0, None)


logging.info('xarrays shapes for '+str(dims_no_apply.keys()) +' + '+str(dims_apply_names)+' : ')
logging.info('  -> original xarrays: '+str(xarrays_shapes))
logging.info('  ->  chunked xarrays: '+str(xarrays_shapes_in_chunks))

maximum_memory_size = 3*10**8
logging.info('determining input chunk format that fits our maximum memory size input of '+str(maximum_memory_size))

chunks_memory_sizes =     [int(xarray.nbytes/xarray.size) for xarray in xarrays+xarrays_out]
chunks_memory_sizes_dim = [[int(xarray.nbytes/xarray.size)] for xarray in xarrays+xarrays_out]


iteration_over_apply_dims = list(reversed(list(enumerate(dims_all))[-len(dims_apply_names):]))
for idim,dimname in iteration_over_apply_dims:
    for ixarray,xarray in enumerate(xarrays+xarrays_out):
        if xarrays_shapes_in_chunks[ixarray][idim] != None:
            chunks_memory_sizes[ixarray] *= xarrays_shapes_in_chunks[ixarray][idim]
            chunks_memory_sizes_dim[ixarray].insert(0,xarrays_shapes_in_chunks[ixarray][idim])


iteration_over_noapply_dims = list(reversed(list(enumerate(dims_all))[:len(dims_no_apply)]))
current_memory_size  = sum(chunks_memory_sizes)

chunk_sizes_no_apply = {}
for idim,dimname in iteration_over_noapply_dims:
    if (current_memory_size < maximum_memory_size):
        chunks_memory_sizes_total = sum(chunks_memory_sizes)
        xarrays_sized_cumulative_base = 0
        xarrays_sized_cumulative_mul = 0
        for ixarray,xarray in enumerate(xarrays+xarrays_out):
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
                chunk_sizes_no_apply[dimname] = np.floor(np.min([(maximum_memory_size - xarrays_sized_cumulative_base)/xarrays_sized_cumulative_mul, dims_no_apply_lengths[dimname]]))
            else:
                chunk_sizes_no_apply[dimname] = 0

        if not chunk_sizes_no_apply[dimname].is_integer():
            raise ValueError('whole number expected for dimension selection size')

        #import pdb; pdb.set_trace()
        current_memory_size = int((xarrays_sized_cumulative_base + xarrays_sized_cumulative_mul * chunk_sizes_no_apply[dimname]))
        #print(current_memory_size > maximum_memory_size)
        #import pdb; pdb.set_trace()
        # if current_memory_size > maximum_memory_size:
        #     raise ValueError('something wrong with the chunk size calculation for limiting memory')
        for ixarray,xarray in enumerate(xarrays+xarrays_out):
            if xarrays_shapes_in_chunks[ixarray][idim] != None:
                chunks_memory_sizes[ixarray] *= chunk_sizes_no_apply[dimname]
                chunks_memory_sizes_dim[ixarray].insert(0,chunk_sizes_no_apply[dimname])
            else:
                chunks_memory_sizes_dim[ixarray].insert(0, None)
        if current_memory_size != sum(chunks_memory_sizes):
            import pdb; pdb.set_trace()
            raise ValueError('inconsistency in de dimension selection size calculation')
    else:
        for ixarray,xarray in enumerate(xarrays+xarrays_out):
            chunks_memory_sizes_dim[ixarray].insert(0,1)
        chunk_sizes_no_apply[dimname] = 1

logging.info('overall memory size: '+ str(current_memory_size))
logging.info('xarray chunk memory sizes for arrays: '+ str(chunks_memory_sizes))
logging.info('xarray chunk memory size per dimension (last one is character byte size): '+ str(chunks_memory_sizes_dim))

chunks_number_no_apply = {}
for dimname,shape in dims_no_apply_lengths.items():
    chunks_number_no_apply[dimname] = np.ceil(dims_no_apply_lengths[dimname]/chunk_sizes_no_apply[dimname])

logging.info('memory input size of chunks: '+ str(current_memory_size) +'/'+str(maximum_memory_size) +' = '+str(current_memory_size/int(maximum_memory_size)*100)+'% of maximum')
if current_memory_size > maximum_memory_size:
    logging.critical('critical. I cannot fix chunks into memory. Please consider using (smaller) chunksize '
                 'for apply output dimensions')


#import pdb; pdb.set_trace()

logging.info('Start looping over chunks and create selection from xarray inputs')

chunks_no_apply = list(product(*tuple([list(range(int(a))) for a in list(chunks_number_no_apply.values())])))
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
            chunk_start.insert(0,int(dim_apply_start * coordinates_output[dimname_apply]['chunksize']))
            chunk_end.insert(0,int(chunk_start[0]+  coordinates_output[dimname_apply]['chunksize']))
            idx_mod -= dim_apply_start*dim_fac
            dim_fac *= number_of_chunks_apply_dims[dimname_apply]

            # chunk_start.insert(0,[])
            # chunk_end.insert(0,[])
            # for ichunk in range(int(chunks_number_no_apply[dimname])):
            #     dim_apply_start = np.mod(idx_mod/dim_fac, number_of_chunks_apply_dims[dimname_apply] )
            #     chunk_start[0].append(int(dim_apply_start * coordinates_output[dimname_apply]['chunksize']))
            #     chunk_end[0].append(int(chunk_start[0]+  coordinates_output[dimname_apply]['chunksize']))
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
        else:
            chunk_start.insert(0,None)
            chunk_end.insert(0,None)

    xarrays_selection_chunk = []
    for ixarray, xarray in enumerate(xarrays+xarrays_out):
        xarrays_selection_chunk.append({})
        for idim,dimname in enumerate(dims_all):
            #dimname = list(dims_no_apply_lengths.keys())[idim]
            if xarrays_shapes_in_chunks[ixarray][idim] is not None:
                if dimname != '__chunk__':
                    if (dimname in dims_apply_names) and \
                            (dimname in coordinates_output) and \
                            ('chunksize' in coordinates_output[dimname]) and \
                            identical_xarrays(xarray[dimname], coordinates_output[dimname]['coordinates']):
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    elif dimname in dims_no_apply.keys():
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    # else:
                    #     xarrays_selection_chunk[ixarray][dimname] = None
        # for idim in range(len(index_no_apply)):
        #     dimname = list(dims_no_apply_lengths.keys())[idim]
        #     if xarrays_shapes_in_chunks[ixarray][idim] is not None:
        #         if dimname != '__chunk__':

        #             if (dimname in dims_apply_names) and \
        #                     (dimname in coordinates_output) and \
        #                     ('chunksize' in coordinates_output[dimname]) and \
        #                     identical_xarrays(xarray[dimname], coordinates_output[dimname]['coordinates']):
        #                 xarrays_selection_chunk[ixarray][dimname] = range(chunk_start,chunk_end)
        #             else:
        #                 xarrays_selection_chunk[ixarray][dimname] = None
    logging.debug('xarray selection: '+str(xarrays_selection_chunk))

                        # number_of_chunks = []
# for ixarray in range(len(xarrays)):
#     number_of_chunks.append(np.array(xarrays_shapes_in_chunks[ixarray],dtype=float)/ np.array(chunks_memory_sizes_dim[ixarray][:-1]))


#    xarrays_shapes_in_chunks_cumulative[ixarray] *= xarrays_shapes_in_chunks[ixarray][0]

#     # we consider different chunking memory sizes for the input is estimated according to different chunking of the current dimension, considering that the corresponding dimensions may have different lengths among the xarrays.
#     dimlengths = np.sort(np.unique([chunks_shapes[ixarray][0] for ixarray in len(xarrays)]))
#     import pdb; pdb.set_trace()
# 
#     # for dimlength in dimlengths:

    


#chunks_shapes[ixarray][0]*chunks_size_cumulative[ixarray][0] for ixarray in xarray

#bordersizes = {'latitude':500,'longitude':500}


# # commented out
# chunks_start_end = {}
# for dimname,chunksize in chunks_shapes.items():
#     #chunks_start = np.array(range(0,len(ds[dimname]),int(len(ds[dimname])/chunksize)))
#     chunks_start = np.array(range(0,len(ds[dimname]),chunksize))
#     chunks_end = chunks_start + chunksize
#     chunks_end[-1] = min(len(ds[dimname]),chunks_end[-1])
#     chunks_start[-1] = chunks_end[-1] - chunksize
#     chunks_start_end[dimname] = list(zip(chunks_start,chunks_end))
# 
# chunks_start_end_combinations = pd.DataFrame(list(product(*chunks_start_end.values())),columns=chunks_start_end.keys())
# 
# 
# chunks = []
# for ichunk,chunk_combination in chunks_start_end_combinations.iterrows():
# #ichunk,chunk_combination = list(chunks_start_end_combinations.iterrows())[0]
# 
#     chunk_ranges = {}
#     for dim,start_end in dict(chunk_combination).items():
#         chunk_ranges[dim] = range(start_end[0],start_end[1])
#     chunks.append(ds.isel(chunk_ranges))
# 


#ds_chunks = []




# a = reclass_array(values_da_arr, [2, 4, 11, 12])
# profile = ds.profile
# profile.update(
#     driver='GTiff',
#     count=1,
#     tiled=True,
#     compress='deflate',
#     dtype='float32',
#     NUM_THREADS='ALL_CPUS',
#     width=ds.shape[1],
#     height=ds.shape[0])
# with RIOFile('test.nc', 'w', **profile) as r_file:
#     da.store(a, r_file, lock=True)

# xrin_coarse = '/projects/C3S_EUBiodiversity/data/case_klimpala/ensemble_merge/interpolated_to_africa_0.5degree/aggregation-30-years/indicators-annual/cropped_to_africa/bias_corrected/cmip5_daily/temperature-daily-maximum_annual_minimum_ensemble-rcp85_median_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_0.5deg_by_0.5deg.nc'






# class RIOFile(object):
#     """Rasterio wrapper to allow da.store to do window saving."""
#     def __init__(self, *args, **kwargs):
#         """Initialize the object."""
#         self.args = args
#         self.kwargs = kwargs
#         self.rfile = None
#     def __setitem__(self, key, item):
#         """Put the data chunk in the image."""
#         if len(key) == 3:
#             indexes = list(range(
#                 key[0].start + 1,
#                 key[0].stop + 1,
#                 key[0].step or 1
#             ))
#             y = key[1]
#             x = key[2]
#         else:
#             indexes = 1
#             y = key[0]
#             x = key[1]
#         chy_off = y.start
#         chy = y.stop - y.start
#         chx_off = x.start
#         chx = x.stop - x.start
#         # band indexes
#         self.rfile.write(item, window=Window(chx_off, chy_off, chx, chy),
#                          indexes=indexes)
#     def __enter__(self):
#         """Enter method."""
#         self.rfile = rasterio.open(*self.args, **self.kwargs)
#         return self
#     def __exit__(self, exc_type, exc_value, traceback):
#         """Exit method."""
#         self.rfile.close()
# def reclass_array(input_array: np.ndarray, value_list):
#     return np.where(np.isin(input_array, value_list), 1, 0)
#