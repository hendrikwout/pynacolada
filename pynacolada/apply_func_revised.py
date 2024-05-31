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
#input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
input_file = '/home/woutersh/projects/KLIMPALA_SF/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
ds = xr.open_dataset( input_file,)['Band1'].rename({'lat':'latitude'}).rename({'lon':'longitude'})
ds = ds.isel(
    latitude  = ( (ds.latitude  > -5) & (ds.latitude  < 5)),
    longitude = ( (ds.longitude > -10) & (ds.longitude < 5))
        )

#input_file = '/projects/C3S_EUBiodiversity/data/case_klimpala/aggregation-30-years/indicators-annual/cropped_to_africa/bias_corrected/cmip5_daily/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
input_file = '/home/woutersh/projects/KLIMPALA_SF/data/test/temperature-daily-mean_annual_mean_IPSL-CM5A-MR_rcp85_r1i1p1_bias-corrected_to_era5_id0daily_1950-01-01_2100-12-31_id0_aggregation-30-year-median_grid_of_IPSL-CM5A-MR_latitude:irregular_longitude:-42.5,65.0,2.5.nc'
ds2 = xr.open_dataarray(input_file)
xarrays = [ds2]

# this also sets the order of (inner) output dimensions as expected by the function
dims_apply_names = ['latitude','longitude']

ignore_memory_limit = False

# this also defines the order in which the output dimensions are being constructed, allowing for transposing the output on the fly.
output_coordinates = { #input
#        'time':{ 'coordinates':ds2.time},
        'longitude':{ 'coordinates':ds.longitude,'chunksize':1000,'overlap':50},
    'latitude':{ 'coordinates':ds.latitude,'chunksize':500,'overlap':50},
        }

xarrays_output_coordinates = [] #default
# xarrays_output_coordinates = [ #input
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


# if output_coordinates is None:
#     output_coordinates_sort_keys = list(output_coordinates.keys())
# else:
#     output_coordinates_sort_keys = []

logging.debug('adding missing output dimensions from specified xarrays_coordinates_output')
if xarrays_output_coordinates != []:
    output_coordinates_new = {}
    for dimdict in xarrays_output_coordinates:
        for dimname in dimdict.keys():
            if dimname not in output_coordinates_new.keys():
                logging.info('adding missing output_coordinates from xarrays_output_coordinates for '+dimname+': '+dimdict[dimname])
                output_coordinates_new[dimname] = dimdict[dimname]
    logging.debug('overriding values and order of previous output_coordinates.')

    if output_coordinates is None:
        output_coordinates = {}

    for output_coordinates_orig in output_coordinates.keys():
        output_coordinates_new[output_coordinates_orig] = output_coordinates[output_coordinates_orig]
    output_coordinates = sort_dict_by_keys(output_coordinates_new,list(output_coordinates.keys()))


output_coordinates_new = {}
for dimname,coordinates in dims_no_apply.items():

    if dimname not in output_coordinates_new.keys():
        logging.info('adding no apply dimensions to the default output_coordinates '+dimname)
        output_coordinates_new[dimname] = {'coordinates':coordinates}

if output_coordinates is None:
    output_coordinates = {}

for output_coordinates_orig in output_coordinates.keys():
    output_coordinates_new[output_coordinates_orig] = output_coordinates[output_coordinates_orig]
output_coordinates = sort_dict_by_keys(output_coordinates_new, list(output_coordinates.keys()))

output_coordinates_new = {}
for dimname in dims_apply_names:
    for xarray in xarrays:
        if dimname in xarray.coords.keys():
            if dimname not in output_coordinates_new.keys():
                logging.info('adding apply dimensions from input xarray to the default output_coordinates ' + dimname)
                output_coordinates_new[dimname] = {'coordinates': xarray[dimname]}

if output_coordinates is None:
    output_coordinates = {}

for output_coordinates_orig in output_coordinates.keys():
    output_coordinates_new[output_coordinates_orig] = output_coordinates[output_coordinates_orig]
output_coordinates = sort_dict_by_keys(output_coordinates_new, list(output_coordinates.keys()))

for dimname in dims_apply_names:
    if dimname not in output_coordinates:
        raise IOError('I cannot track the dimension size and coordinates of '+dimname+' please specify in output_coordinates')


    # for ixarray_output,xarray_output_coordinates in xarrays_output_coordinates:
    #     for dimname,coordinates in xarray_output_coordinates.items():
    #         if dimname not in output_coordinates.keys():
    #             output_coordinates[dimname] = xarray_output_coordinates[ixarray_output][dimname]

number_of_chunks_apply = 1
number_of_chunks_apply_dims = {}
for dimname,dimattr in output_coordinates.items():
    if 'chunksize' in output_coordinates[dimname]:
        number_of_chunks_apply_dims[dimname] = int(np.ceil(len(output_coordinates[dimname]['coordinates'])/output_coordinates[dimname]['chunksize']))
        number_of_chunks_apply *= number_of_chunks_apply_dims[dimname] #np.ceil(len(output_coordinates[dimname]['coordinates'])/output_coordinates[dimname]['chunksize'])

number_of_chunks_apply = int(number_of_chunks_apply)

logging.info('Adding extra chunking as a separate dimension')
if number_of_chunks_apply > 1:
    dims_no_apply['__chunk__'] = xr.DataArray(range(number_of_chunks_apply))

dims_all = list(dims_no_apply.keys()) + dims_apply_names
dims_no_apply_lengths = { name:dim.shape[0] for (name,dim) in dims_no_apply.items()}
# dims_apply_shapes = { name:dim.shape for (name,dim) in dims_apply_names.items()}
#dims_all_shapes = {**dims_no_apply_shapes,**dims_apply_shapes}


# for dimname,coordinates in dims_no_apply.items():
#     if dimname not in output_coordinates.keys():
#         logging.info('adding no apply dimensions to the default output_coordinates '+dimname+': '+coordinates)
#         output_coordinates[dimname] = {'coordinates':coordinates}

xarrays_output_filenames = ['/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing.nc','/home/woutersh/projects/KLIMPALA_SF/data/test_output/testing2.nc']

while (len(xarrays_output_coordinates) < len(xarrays_output_filenames) ):
    logging.info('No coordinates output xarrays are set manually, so we guess them from the output_coordinates.'
                 'We do this here already so that we can take it into account in the memory size and optimal chunking.')
    xarrays_output_coordinates.append(output_coordinates)

if len(xarrays_output_filenames) != len(xarrays_output_coordinates):
    raise IOError('number of output files are not the same as the number of expected output xarrays')


xarrays_output_filenames_work = []
directory_work = '/tmp/'

logging.info('Setting working output files')

xarrays_out = []

if xarrays_output_filenames is not None:
    ncouts = []
    xarrays_output_filenames_work = []
    for filename_out in xarrays_output_filenames:
        if not directory_work:
            xarrays_output_filenames_work.append(filename_out)
            logging.info("Dump output directly to final destination: " + xarrays_output_filenames_work[-1])
        else:
            logging.info("Using temporary output dir, eg., good for working with network file systems")
            if (directory_work is None) or (directory_work is True):
                xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=None))
                logging.info("Using temporary output in default directory_work: " + xarrays_output_filenames_work[-1])
            else:
                xarrays_output_filenames_work.append(tempfile.mktemp(suffix='.nc', dir=directory_work))
                logging.info("Using temporary output in specified directory_work: " + xarrays_output_filenames_work[-1])

    xrtemp = xr.Dataset()
    for ixarray_out in range(len(xarrays_output_coordinates)):
        for dimname, dim in xarrays_output_coordinates[ixarray_out].items():
            xrtemp[dimname] = dim['coordinates']
            # ncouts[iarray].createDimension(dim,shapes_out_transposed[iarray][idim])
            # ncouts[iarray].createVariable(dim,'d',(dim,),)
            # ncouts[iarray].variables[dim][:] = coords_out_transposed[iarray][idim]
        fnout = xarrays_output_filenames_work[ixarray_out]  # 'testing_'+str(iarray)+'.nc'
        if os.path.isfile(fnout):
            raise IOError('output file ' + fnout + ' exists. Aborting... ')
        # os.system('rm ' + fnout)
        xrtemp.to_netcdf(fnout)
        xrtemp.close()
        ncouts.append(nc4.Dataset(fnout, 'a'))
        ncouts[ixarray_out].createVariable('__xarray_data_variable__', "f", tuple(xarrays_output_coordinates[ixarray_out].keys()),fill_value=0.)
        #ncouts[ixarray_out].variables['__xarray_data_variable__'][:] = 0.
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
                    (dimname in output_coordinates) and
                    ('chunksize' in list(output_coordinates[dimname].keys())) and
                    identical_xarrays(xarray[dimname],output_coordinates[dimname]['coordinates'])
               ):
                xarrays_shapes_in_chunks[ixarray].insert(0,output_coordinates[dimname]['chunksize'])
                if 'overlap' in output_coordinates[dimname]:
                    xarrays_shapes_in_chunks[ixarray][0] += output_coordinates[dimname]['overlap']

                xarrays_chunks_apply[ixarray] = True
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray[dimname]))
            xarrays_shapes[ixarray].insert(0,xarray.shape[xarray.dims.index(dimname)])
        else:
            xarrays_shapes_in_chunks[ixarray].insert(0,None)
            xarrays_shapes[ixarray].insert(0,None)


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

maximum_memory_size = 2*10**7
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

    logging.critical('expected memory usage exceeds predefined memory limit. \n'+\
                          ' - expected memory usage: '+ str(current_memory_size)+ '\n' + \
                          ' - limit of memory usage: '+ str(current_memory_size)+ '\n' + \
                          ' - chunks_memory_sizes: '+str(chunks_memory_sizes)+ '\n'+ \
                          ' - chunks_memory_sizes_dim'+str(chunks_memory_sizes)+ '\n'+ \
                          'Please consider the usage of memory chunking along the apply_dimensions'
                          )
    if not ignore_memory_limit:
        raise IOError('memory limit needs to be respected. Or turn on ignore_memory_linit')

#import pdb; pdb.set_trace()

logging.info('Start looping over chunks and create selection from xarray inputs')

chunks_no_apply = list(product(*tuple([list(range(int(a))) for a in list(chunks_number_no_apply.values())])))


logging.debug('closing xarrays_out, which we used above to easily calculate the dimension/memory shaping. Data is written out directly through disk for saving memory. ')
for ixarray_out, xarray_out in enumerate(xarrays_out):
    xarrays_out[ixarray_out].close()
    ncouts[ixarray_out] = nc4.Dataset(xarrays_output_filenames_work[ixarray_out],'a')


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
            chunk_start.insert(0,int(dim_apply_start * output_coordinates[dimname_apply]['chunksize']))
            chunk_end.insert(0,int(chunk_start[0] +  output_coordinates[dimname_apply]['chunksize']))
            if 'overlap' in output_coordinates[dimname_apply]:
                chunk_end[0] += output_coordinates[dimname_apply]['overlap']
            chunk_end[0] = np.min([chunk_end[0],len(output_coordinates[dimname_apply]['coordinates'])])
            idx_mod -= dim_apply_start*dim_fac
            dim_fac *= number_of_chunks_apply_dims[dimname_apply]

            # chunk_start.insert(0,[])
            # chunk_end.insert(0,[])
            # for ichunk in range(int(chunks_number_no_apply[dimname])):
            #     dim_apply_start = np.mod(idx_mod/dim_fac, number_of_chunks_apply_dims[dimname_apply] )
            #     chunk_start[0].append(int(dim_apply_start * output_coordinates[dimname_apply]['chunksize']))
            #     chunk_end[0].append(int(chunk_start[0]+  output_coordinates[dimname_apply]['chunksize']))
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
            chunk_end[0] = np.min([chunk_end[0],len(output_coordinates[dimname]['coordinates'])])
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
                            (dimname in output_coordinates) and \
                            ('chunksize' in output_coordinates[dimname]) and \
                            identical_xarrays(xarray[dimname], output_coordinates[dimname]['coordinates']):
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    elif dimname in dims_no_apply.keys():
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    else:
                        xarrays_selection_chunk[ixarray][dimname] = range(0,xarrays_shapes_in_chunks[ixarray][idim])

    xarrays_selection_chunk_in = xarrays_selection_chunk[:len(xarrays)]
    xarrays_selection_chunk_out = xarrays_selection_chunk[len(xarrays):len(xarrays+xarrays_out)]
    logging.debug('xarrays selection in  : '+str(xarrays_selection_chunk_in))
    logging.debug('xarrays selection out : '+str(xarrays_selection_chunk_out))

    chunks_in = []
    coordinates_in = []
    for ixarray,xarray in enumerate(xarrays):

        chunks_in.append(xarray.isel(xarrays_selection_chunk_in[ixarray]).transpose(*tuple(xarrays_selection_chunk_in[ixarray].keys())))

    dims_not_found = {}
    for dim,coordinate_output in output_coordinates.items():
        dimfound = False
        for ixarray,xarray in enumerate(xarrays):
            if (dim in xarray.coords) and (identical_xarrays(coordinate_output['coordinates'],xarray.coords[dim])):
                dimfound = True

        if dimfound == False:
            dims_not_found[dim] = output_coordinates[dim]['coordinates'].isel(
                {dim:range(chunk_start[dims_all.index(dim)], chunk_end[dims_all.index(dim)])}

            )
    #chunk_output_coordinates = output_coordinates[dim].isel(art})
    if len(dims_not_found ) > 0:
        logging.info('output coordinates that are missing in the input files are found. We will pass them to the function:' + str(dims_not_found))

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
    def func (x_coarse,**coordinates_fine):
        out = (pcd.vectorized_functions.extend_crop_interpolate( \
            x_coarse.values, \
            (x_coarse.latitude.values,x_coarse.longitude.values), \
            (coordinates_fine['latitude'].values,coordinates_fine['longitude'].values), \
            # interpolation=True,
            # return_grid_output=False,
            # debug=False,
            # border_pixels=5,
            # ascending_lat_lon = False,
            # tolerance_for_grid_match = 1.e-9
        ))
        return out,out


    chunks_out = func(*chunks_in,**dims_not_found)

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

        xarrays_selection_chunk_out_ordered = sort_dict_by_keys(xarrays_selection_chunk_out[ixarray_out],list(xarrays_output_coordinates[ixarray_out].keys()))
        #if type(chunk_out) == type(np.array([])):
        if type(chunk_out) != xr.core.dataarray.DataArray:
            chunk_out_coordinates = {}
            for dimname in xarrays_selection_chunk_out[ixarray_out].keys():
                chunk_out_coordinates[dimname] = xarrays_output_coordinates[ixarray_out][dimname]['coordinates'].isel({dimname:xarrays_selection_chunk_out[ixarray_out][dimname]})

            chunk_out_xarray = xr.DataArray(chunk_out,coords=chunk_out_coordinates)
        else:
            chunk_out_xarray = chunk_out

        logging.debug('xarray selection ordered for output array '+str(ixarray_out)+': ' + str(xarrays_selection_chunk_out_ordered))


        chunk_out_xarray_ordered = chunk_out_xarray.transpose(*tuple(xarrays_selection_chunk_out_ordered.keys()))
        # chunk_profile =

        logging.debug('re-ordered output shape: '+str(chunk_out_xarray.shape) +' -> '+ str(chunk_out_xarray_ordered.shape))
        indexing_for_output_array = tuple([dim_selection for dim_selection in xarrays_selection_chunk_out_ordered.values()])
        logging.debug('index of chunk in netcdf output '+str(ixarray_out)+': ' + str(indexing_for_output_array))
        logging.debug('this should fit in netcdf total output shape '+str(ncouts[ixarray_out].variables['__xarray_data_variable__'].shape))

        logging.debug('acquiring previous values for consolidating chunk overlapping values')
        test = ncouts[ixarray_out].variables['__xarray_data_variable__'][indexing_for_output_array].filled(fill_value=0)

        overlap_weights = np.ones_like(test)
        idim = 0
        for dim,selection_chunk_out in xarrays_selection_chunk_out_ordered.items():
            # overlap_weights_dim = np.ones((len(xarrays_selection_chunk_out_ordered[dim],)))
            #reshape(list(range(idim-1))+overlap_weights.shape[idim])
            if 'overlap' in output_coordinates[dim]:
                if xarrays_selection_chunk_out_ordered[dim][0] == 0:
                    left = np.ones(output_coordinates[dim]['overlap'])
                else:
                    left = np.arange(0,output_coordinates[dim]['overlap'],1)/output_coordinates[dim]['overlap']
                if xarrays_selection_chunk_out_ordered[dim][-1] == (len(output_coordinates[dim]['coordinates']) - 1):
                    right = np.ones(output_coordinates[dim]['overlap'])
                else:
                    right = np.arange(output_coordinates[dim]['overlap'],0,-1)/output_coordinates[dim]['overlap']
                middle = np.ones(((len(xarrays_selection_chunk_out_ordered[dim]) - 2 * output_coordinates[dim]['overlap'],)))
                overlap_weights_dim = np.concatenate([left,middle,right])
                overlap_weights_dim = overlap_weights_dim.reshape([1]*idim+[overlap_weights.shape[idim]]+[1]*(len(overlap_weights.shape) - idim -1))
                overlap_weights *= overlap_weights_dim
            idim += 1

        ncouts[ixarray_out].variables['__xarray_data_variable__'][indexing_for_output_array] = test + np.array(chunk_out_xarray_ordered.values,dtype='float32')*overlap_weights
        logging.debug('.... finished')
    for ichunk_out in range(len(chunks_out)):
        if type(chunks_out[ichunk_out]) == xr.core.dataarray.DataArray:
            chunks_out[ichunk_out].close()

        #del chunks_out[ichunk_out]
        # import pdb; pdb.set_trace()

for incout in range(len(ncouts)):
    ncouts[incout].close()
    os.system('mv '+xarrays_output_filenames_work[incout]+' '+xarrays_output_filenames[incout])
            # else:
            #     xarrays_selection_chunk[ixarray_out][dimname] = None
            # for idim in range(len(index_no_apply)):
            #     dimname = list(dims_no_apply_lengths.keys())[idim]
            #     if xarrays_shapes_in_chunks[ixarray_out][idim] is not None:
        #         if dimname != '__chunk__':

        #             if (dimname in dims_apply_names) and \
        #                     (dimname in output_coordinates) and \
        #                     ('chunksize' in output_coordinates[dimname]) and \
        #                     identical_xarrays(xarray[dimname], output_coordinates[dimname]['coordinates']):
        #                 xarrays_selection_chunk[ixarray_out][dimname] = range(chunk_start,chunk_end)
        #             else:
        #                 xarrays_selection_chunk[ixarray][dimname] = None
    # import pdb; pdb.set_trace()

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
