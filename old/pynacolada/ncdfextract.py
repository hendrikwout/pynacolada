import datetime as dt
import numpy as np

def ncgettypecode(dtype):
    ''' purpose: netcdf-typecode from array-dtype
    '''
    if ((dtype == np.dtype('float32')) or (dtype == 'float32')):
        return 'f'
    elif ((dtype == np.dtype('float64')) or (dtype == 'float64')):
        return 'd'
    elif ((dtype == np.dtype('int32')) or (dtype == 'int32')):
        return 'i'
    elif ((dtype == np.dtype('int64')) or (dtype == 'int64')):
        return 'l'

def ncgetdatetime(ncin):
    ''' extract datetimes from the 'time' coordinate in ncin
        ncin: input netcdf file
        returns an array of datetimes
    '''
    tunits = getattr(ncin.variables['time'],'units').split()
    if tunits[0] == 'days':
        mul = 24.*3600.
    elif tunits[0] == 'hours':
        mul = 3600.
    elif tunits[0] == 'minutes':
        mul = 60.
    elif tunits[0] == 'seconds':
        mul = 1.
    else:
        raise Exception("no time conversion found for '"+tunits[0],+"'")
    
    try:
        refdat = dt.datetime.strptime(tunits[2]+' '+tunits[3],"%Y-%m-%d %H:%M:%S")
    except:
        refdat = dt.datetime.strptime(tunits[2]+' '+tunits[3],"%Y-%m-%d %H:%M")
    
    return [(refdat+ dt.timedelta(seconds=e*mul)) for e in ncin.variables['time'][:]]
    #return(multifunc(lambda x: refdat + timedelta(seconds=x[0]), [ncin.variables['time'][:]*mul],[False],[[0]])[0])


