import dask.array as da
import numpy as np
import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from rasterio.windows import Window
ProgressBar().register()
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


input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
ds = xr.open_dataset(
    input_file,
)['Band1']

chunksize = 500

dims_apply = ['latitude','longitude']
chunksizes = {'latitude':500,'longitude':500}
bordersizes = {'latitude':500,'longitude':500}

#bordersize = 10

chunks = {}
for dimname,chunksize in chunksizes.items():
    chunks[dimname] = list(range(0,ds['dimname'],chunksize/ds['dimname']))j

# for chunk in range(chunksize):
#     ds.isel({'lat':range(0,500),'lon':range(0,500)})

# input_file = '/projects/C3S_EUBiodiversity/data/ancillary/GMTED2010/gmted2010_mean_30.nc'
# ds = xr.open_dataset(
#     input_file,
#     chunks={
#         'lat':int(ds.shape[0] / 40),
#         'lon':int(ds.shape[1] / 40)}
# )['Band1']

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






