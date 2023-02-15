import numpy as np
import os
import xarray as xr
import pandas as pd
#from tabulate import tabulate
import glob
#from xarray_extras import csv
from .io_gsod import GSOD as gsod
import logging

def get_argparse_from_function(function):

    import argparse
    import inspect

    # problem! if you have reset_archive_after then the reset_archive argument disappears
    arguments = {}
    help_dict = {}
    logging.debug('acquiring help information of arguments')
    for help_line in function.__doc__.split('\n'):
        key = help_line.split(':')[0].strip(' ')
        if key != '':
            if ':' in help_line:
                value = ':'.join(help_line.split(':')[1:])
            else:
                value = None
            if key in help_dict.keys():
                logging.warning('help for "'+key+'" already defined. \n Overwriting "'+str(help_dict[key])+'" with '+str(value))
            help_dict[key] = value


    logging.debug('Setting parser for function arguments')
    parser = argparse.ArgumentParser(function.__doc__)
    for parkey,parvalue in inspect.signature(function).parameters.items():
        if parkey not in ['args','kwargs']:
            arguments[parkey] = {}
            if parvalue.default is not inspect._empty:
                arguments[parkey]['default'] = parvalue.default
            if parvalue.annotation is not inspect._empty:
                arguments[parkey]['type'] = parvalue.annotation
            if (parkey in help_dict.keys()) and (help_dict.keys() is not None):
                arguments[parkey]['help'] = help_dict[parkey]

            parser.add_argument('--'+parkey,**arguments[parkey])

    if 'kwargs' in inspect.signature(function).parameters:
        parsed, unknown = parser.parse_known_args()
        arg_keys_propagate = []
        for arg in unknown:
            if arg.startswith(("-", "--")):
                # you can pass any arguments to add_argument
                parser.add_argument(arg.split('=')[0], type=str)
                arg_keys_propagate.append(arg.split('=')[0].strip('-'))
    return parser


def DataArray_like(da,name=None,**kwargs):
    da_out = da.copy()
    da_out.values[:] = 0.
    if name is not None:
        da_out.name = name
    for key,value in kwargs.items():
        da_out.attrs[key] = value
    return da_out

def dataarray_to_ascii(xarray,fname,fmt='.5E',nanvalue=-9.99999E+20):
    '''
    purpose: put a 2d data-array into gis-type ascii format.
    '''
    if type(fname).__name__ == 'str':
        writefile = open(fname, "w")
    else:
        writefile = fname
    if len(xarray.shape) != 2:
            raise ValueError('data array dimension '+str(len(xarray.shape))+' is different from 2.')

    #print(df_series.columns[0],df_series.columns[1] ,df_series.index[-1] ,df_series.index[-2])
#     import pdb;pdb.set_trace()
    writefile.write('ncols          '+str(len(xarray.longitude))+'\n')
    writefile.write('nrows          '+str(len(xarray.latitude))+'\n')
    writefile.write('xllcorner      '+str(xarray.longitude[0].values - (xarray.longitude[1].values - xarray.longitude[0].values)/2.)+'\n')
    writefile.write('yllcorner      '+str(xarray.latitude[0].values - (xarray.latitude[1].values - xarray.latitude[0].values)/2.)+'\n')
    writefile.write('cellsize       '+str(xarray.latitude[1].values-xarray.latitude[0].values)+'\n')
    writefile.write('NODATA_value   nan'+'\n')
    #xarray.series_to_fwf(writefile)
    np.savetxt(writefile,xarray.values.squeeze()[::-1],fmt='%'+fmt)#'%.5E')

    if ((nanvalue is not None) and (type(nanvalue).__name__ != 'str')):
        writefile.close()
        if np.sum(xarray.values.ravel() == nanvalue) >= 1:
            raise ValueError('nan value '+format(nanvalue,fmt)+' occurs as actual number value in the dataarray. Please choose another nanvalue.')
        #import pdb;pdb.set_trace()
        os.system("sed 's/nan/"+format(nanvalue,fmt)+"/g' -i "+writefile.name)
        writefile = open(writefile.name,'a')
    elif ((nanvalue is not None) and (type(nanvalue).__name__ == 'str')):
        os.system("sed 's/nan/"+nanvalue+"/g' -i "+writefile)
        writefile.close()
        writefile = open(writefile.name,'a')

    if type(fname).__name__ == 'str':
        writefile.close();print('file written to: ',writefile)

# def series_to_fwf(df_series, fname,show_columns=False):
