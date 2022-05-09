#import dask.array as da
from itertools import product
import numpy as np
#import rasterio
import xarray as xr
#from dask.diagnostics import ProgressBar
#from rasterio.windows import Window
import pandas as pd
import numpy as np
import logging
#ProgressBar().register()
class RIOFile(object):
    """Rasterio wrapper to allow da.store to do window saving."""
    def __init__(self, *args, **kwargs):
        """Initialize the object."""
        self.args = args
        self.kwargs = kwargs
        self.rfile = None
    def __setitem__(self, key, item):
        """Put the data chunk in the image."""
        if len(key) == 3:
            indexes = list(range(
                key[0].start + 1,
                key[0].stop + 1,
                key[0].step or 1
            ))
            y = key[1]
            x = key[2]
        else:
            indexes = 1
            y = key[0]
            x = key[1]
        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start
        # band indexes
        self.rfile.write(item, window=Window(chx_off, chy_off, chx, chy),
                         indexes=indexes)
    def __enter__(self):
        """Enter method."""
        self.rfile = rasterio.open(*self.args, **self.kwargs)
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        self.rfile.close()
def reclass_array(input_array: np.ndarray, value_list):
    return np.where(np.isin(input_array, value_list), 1, 0)
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

dims_apply_names = ['latitude','longitude']


dims_output = {
        'latitude':{ 'coordinates':ds.latitude,'chunksize':500},
        'longitude':{ 'coordinates':ds.longitude,'chunksize':500},
        }

logging.debug('determining inner number of chunks')

number_of_chunks_apply = 1
number_of_chunks_apply_dims = {}
for dimname,dimattr in dims_output.items():
    if 'chunksize' in dims_output[dimname]:
        number_of_chunks_apply_dims[dimname] = int(np.ceil(len(dims_output[dimname]['coordinates'])/dims_output[dimname]['chunksize']))
        number_of_chunks_apply *= number_of_chunks_apply_dims[dimname] #np.ceil(len(dims_output[dimname]['coordinates'])/dims_output[dimname]['chunksize'])

number_of_chunks_apply = int(number_of_chunks_apply)

def identical_xarrays(xarray1,xarray2):
   return (xarray1.shape == xarray2.shape) and (not (np.any(xarray1 != xarray2))) 

logging.debug('collecting the dimensions occuring in the xarrays over which the function is not applied')
dims_no_apply  = {}

logging.debug('Adding extra chunking as a separate dimension')
if number_of_chunks_apply > 1:
    dims_no_apply['__chunk__'] = xr.DataArray(range(number_of_chunks_apply))

for ixarray,xarray in enumerate(xarrays):
    for dimname in xarray.dims:
        if (dimname not in dims_no_apply) and (dimname not in dims_apply_names):
            dims_no_apply[dimname] = xarray[dimname]
        if (dimname in dims_no_apply) and (not identical_xarrays(xarray[dimname],dims_no_apply[dimname])):
            raise ValueError('dimension '+dimname+' of xarray number '+str(ixarray)+' is not the same as previously detected dimensions.')



dims_all = list(dims_no_apply.keys()) + dims_apply_names
dims_no_apply_lengths = { name:dim.shape[0] for (name,dim) in dims_no_apply.items()}
# dims_apply_shapes = { name:dim.shape for (name,dim) in dims_apply_names.items()}
#dims_all_shapes = {**dims_no_apply_shapes,**dims_apply_shapes}
logging.debug('determining chunked shapes for each xarray')

xarrays_shapes_in_chunks = [list() for i in range(len(xarrays))]
xarrays_shapes = [list() for i in range(len(xarrays))]
#xarrays_shapes_in_chunks_cumulative = [1]*len(xarrays)
#xarrays_dims_transposed = [[]]*len(xarrays)

xarrays_chunks_apply = [False]*len(xarrays)
for ixarray,xarray in enumerate(xarrays):
    for idim,dimname in reversed(list(enumerate(dims_apply_names))):
        print(dimname,ixarray,xarrays_shapes_in_chunks)
        if dimname in xarray.dims:
            if ( 
                    (dimname in dims_output) and 
                    ('chunksize' in list(dims_output[dimname].keys())) and 
                    identical_xarrays(xarray[dimname],dims_output[dimname]['coordinates'])
               ):
                xarrays_shapes_in_chunks[ixarray].insert(0,dims_output[dimname]['chunksize'])
                xarrays_chunks_apply[ixarray] = True
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray[dimname]))
            xarrays_shapes[ixarray].insert(0,xarray.shape[xarray.dims.index(dimname)])
        else:
            xarrays_shapes_in_chunks[ixarray].insert(0,None)
            xarrays_shapes[ixarray].insert(0,None)




for idim,dimname in reversed(list(enumerate(dims_no_apply))):
    for ixarray,xarray in enumerate(xarrays):
        if dimname != '__chunk__': #inner extra chunk dimension is already considered in previous loop
            if dimname in xarray.dims:
                xarrays_shapes_in_chunks[ixarray].insert(0,len(xarray[dimname]))
                xarrays_shapes[ixarray].insert(0, len(xarray[dimname]))
            else:
                xarrays_shapes_in_chunks[ixarray].insert(0,None)
                xarrays_shapes[ixarray].insert(0, None)

logging.info('adding size of __chunk__ dimensions')

if number_of_chunks_apply > 1:
    for ixarray,xarray in enumerate(xarrays):
        if xarrays_chunks_apply[ixarray] == True:
            xarrays_shapes_in_chunks[ixarray].insert(0,number_of_chunks_apply)
        else:
            xarrays_shapes_in_chunks[ixarray].insert(0,None)
        xarrays_shapes[ixarray].insert(0,None)

logging.debug('chunked shapes for '+str(dims_all)+': '+str(xarrays_shapes_in_chunks))

maximum_memory_size_input = 10**8
logging.info('determining input chunk format that fits our maximum memory size input of '+str(maximum_memory_size_input))

chunks_memory_sizes = [int(xarray.nbytes/xarray.size) for xarray in xarrays]
chunks_memory_sizes_dim = [[int(xarray.nbytes/xarray.size)] for xarray in xarrays]
iteration_over_apply_dims = list(reversed(list(enumerate(dims_all))[-len(dims_apply_names):]))
for idim,dimname in iteration_over_apply_dims:
    for ixarray,xarray in enumerate(xarrays):
        if xarrays_shapes_in_chunks[ixarray][idim] != None:
            chunks_memory_sizes[ixarray] *= xarrays_shapes_in_chunks[ixarray][idim]
            chunks_memory_sizes_dim[ixarray].insert(0,xarrays_shapes_in_chunks[ixarray][idim])

iteration_over_noapply_dims = list(reversed(list(enumerate(dims_all))[:len(dims_no_apply)]))
current_memory_size  = sum(chunks_memory_sizes)

chunk_sizes_no_apply = {}
for idim,dimname in iteration_over_noapply_dims:
    if (current_memory_size < maximum_memory_size_input):
        chunks_memory_sizes_total = sum(chunks_memory_sizes)
        xarrays_sized_cumulative_base = 0
        xarrays_sized_cumulative_mul = 0
        for ixarray,xarray in enumerate(xarrays):
            #xarrays_sizes_cumulative[ixarray] *= xarrays_shapes_in_chunks[ixarray][idim]
            if xarrays_shapes_in_chunks[ixarray][idim] == None:
                xarrays_sized_cumulative_base += chunks_memory_sizes[ixarray]
            else:
                xarrays_sized_cumulative_mul += chunks_memory_sizes[ixarray]

        if xarrays_sized_cumulative_mul  != 0:
            chunk_sizes_no_apply[dimname] = np.floor(np.min([(maximum_memory_size_input - xarrays_sized_cumulative_base)/xarrays_sized_cumulative_mul, dims_no_apply_lengths[dimname]]))
        else:
            chunk_sizes_no_apply[dimname] = 0


        if not chunk_sizes_no_apply[dimname].is_integer():
            raise ValueError('whole number expected for dimension selection size')

        for ixarray,xarray in enumerate(xarrays):
            if xarrays_shapes_in_chunks[ixarray][idim] != None:
                chunks_memory_sizes[ixarray] *= chunk_sizes_no_apply[dimname]
                chunks_memory_sizes_dim[ixarray].insert(0,chunk_sizes_no_apply[dimname])
            else:
                chunks_memory_sizes_dim[ixarray].insert(0, None)
        current_memory_size = int((xarrays_sized_cumulative_base + xarrays_sized_cumulative_mul * chunk_sizes_no_apply[dimname]))
        if current_memory_size != sum(chunks_memory_sizes):
            raise ValueError('inconsistency in de dimension selection size calculation')


    else:
        for ixarray,xarray in enumerate(xarrays):
            chunks_memory_sizes_dim[ixarray].insert(0,1)
        chunk_sizes_no_apply[dimname] = 1

chunks_number_no_apply = {}
for dimname,shape in dims_no_apply_lengths.items():
    chunks_number_no_apply[dimname] = np.ceil(dims_no_apply_lengths[dimname]/chunk_sizes_no_apply[dimname])

logging.info('Start looping over chunks and create selection from xarray inputs')

chunks_no_apply = list(product(*tuple([list(range(int(a))) for a in list(chunks_number_no_apply.values())])))
for index_no_apply in chunks_no_apply:

    chunk_start = []
    chunk_end = []

    if '__chunk__' in chunks_number_no_apply.keys():
        dim_fac = 1
        idx_mod = index_no_apply[list(chunks_number_no_apply.keys()).index('__chunk__')]
    for dimname_apply in list(reversed(dims_apply_names)):
        if dimname_apply in number_of_chunks_apply_dims.keys():
            dim_apply_start = np.mod(idx_mod/dim_fac, number_of_chunks_apply_dims[dimname_apply] )
            chunk_start.insert(0,dim_apply_start * dims_output[dimname_apply]['chunksize'])
            chunk_end.insert(0,chunk_start[0]+  dims_output[dimname_apply]['chunksize'])
            idx_mod -= dim_apply_start*dim_fac
            dim_fac *= number_of_chunks_apply_dims[dimname_apply]
        else:
            chunk_start.insert(0,None)
            chunk_end.insert(0,None)

    for idim in reversed(range(len(index_no_apply))):
        dimname = list(dims_no_apply_lengths.keys())[idim]
        if dimname != '__chunk__':
            chunk_start.insert(0,chunk_sizes_no_apply[dimname] * index_no_apply[idim])
            chunk_end.insert(0,chunk_start[0] + chunk_sizes_no_apply[dimname])

        else:
            chunk_start.insert(0,None)
            chunk_end.insert(0,None)

    xarrays_selection_chunk = []
    for ixarray, xarray in enumerate(xarrays):
        xarrays_selection_chunk.append({})
        for idim in range(len(dims_all)):
            dimname = list(dims_no_apply_lengths.keys())[idim]
            if xarrays_shapes_in_chunks[ixarray][idim] is not None:
                if dimname != '__chunk__':
                    if (dimname in dims_apply_names) and \
                            (dimname in dims_output) and \
                            ('chunksize' in dims_output[dimname]) and \
                            identical_xarrays(xarray[dimname], dims_output[dimname]['coordinates']):
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    elif dimname in dims_no_apply.keys():
                        xarrays_selection_chunk[ixarray][dimname] = range(chunk_start[idim],chunk_end[idim])
                    else:
                        xarrays_selection_chunk[ixarray][dimname] = None
        # for idim in range(len(index_no_apply)):
        #     dimname = list(dims_no_apply_lengths.keys())[idim]
        #     if xarrays_shapes_in_chunks[ixarray][idim] is not None:
        #         if dimname != '__chunk__':

        #             if (dimname in dims_apply_names) and \
        #                     (dimname in dims_output) and \
        #                     ('chunksize' in dims_output[dimname]) and \
        #                     identical_xarrays(xarray[dimname], dims_output[dimname]['coordinates']):
        #                 xarrays_selection_chunk[ixarray][dimname] = range(chunk_start,chunk_end)
        #             else:
        #                 xarrays_selection_chunk[ixarray][dimname] = None
    import pdb; pdb.set_trace()

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






