from sciproc import multifunc, procdf
from numpy import *

# from sciproc import *
import inspect
from operator import itemgetter
from numpy import *

def nccopydimension (ncin,ncout,dimension,copyvalues=True):
    """Copy a dimension. By default. The variable named after the dimension is also copied """
    if (dimension in ncin.dimensions) & (dimension not in ncout.dimensions):
        ncout.createDimension(dimension,ncin.dimensions[dimension])
        if (dimension in ncin.variables) & (dimension not in ncout.variables):
            nccopyvariable(ncin,ncout,dimension,copyvalues=copyvalues)
                     # dimoutsize =  indimncpointers[idim].dimensions[edim] 
                     # if dimoutsize == None:
                     #     lvarfound = False
                     #     for evar in indimncpointers[idim].variables:
                     #         if ((lvarfound == False) & (edim in indimncpointers[idim].variables[evar].dimensions)):
                     #             dimoutsize = indimncpointers[idim].variables[evar].shape[indimncpointers[idim].variables[evar].dimensions.index(edim)]
                     #             lvarfound = True



# input:
    # function: the function that has to be applied on the netcdf-files
    # seldimensions: a list of dimension(s) along which the function is being applied
    # ncin: taken into account when nclist == None:
        # a netcdf reference or list which will be treated in a 'smart' way to apply the function
    # copynonrel: make a simply copy of netcdf variables that are not relevant to the function

def nctypecode(array):
    if array.dtype == dtype('float32'):
        return 'f'
    elif array.dtype == dtype('float64'):
        return 'd'
    elif array.dtype == dtype('int32'):
        return 'i'
    elif array.dtype == dtype('int64'):
        return 'l'

def ncshowattrfile(ncin):
    for attr in dir(ncin):
        atvalue = getattr(ncin,attr)
        if (attr not in ['close', 'createDimension', 'createVariable', 'flush', 'sync']):
            print(attr,atvalue,type(atvalue))

def nccopyattrfile(ncin,ncout):
    for attr in dir(ncin):
        atvalue = getattr(ncin,attr)
        # unfortunately, 'setattr' cannot set array/list-type attributes
        if ((type(atvalue).__name__ != 'ndarray') & \
            (attr not in ['close', 'createDimension', 'createVariable', 'flush', 'sync'])):
            setattr(ncout,attr,atvalue)

def nccopyattrvar(ncin,ncout,varin=None,varout=None):
    if varin == None:
        # copy attributes of all variables
        selvar = []
        for evar in ncin.variables:
            selvar.append(evar)
    elif (type(varin).__name__ == 'str'):
        # make a list of the string
        selvar = [varin]
    else:
        # just use the given variables
        selvar = varin

    if varout == None:
        # take the same names as the input variables
        selvarout = selvar
    elif (type(varout).__name__ == 'str'):
        selvarout = [varout]
    else:
        selvarout = varout

    # print 'selvar: ', selvar
    # print 'selvarout: ', selvarout
    for ivar,evarout in enumerate(selvarout):
        if ((evarout in ncout.variables) & (selvar[ivar] in ncin.variables)):
            for attr in dir(ncin.variables[selvar[ivar]]):
                if attr not in ['assignValue', 'getValue', 'typecode']:
                    try:
                        setattr(ncout.variables[evarout],attr,getattr(ncin.variables[selvar[ivar]],attr))
                    except:
                        print('Warning: something went wrong when transferring attributes')

def nccopyvariable(ncin,ncout,varin,varout=None,copyvalues=False,copyattr=True):
    ''' create a new netcdf variable with the same dimensions and attributes as the original variable
        ncin: input netcdf file
        ncout output netcdf file
        varin: input variable
        varout (optional): output variable. If None, it is equal to varin
        copyvalues: copy values (only possible if the dimensions of source and target variable are of the same size)
    '''
    if varout == None:
        varoutdef = varin
    else:
        varoutdef = varout

    if varoutdef not in ncout.variables:
        for edimension in ncin.variables[varin].dimensions:
            if edimension not in ncout.dimensions:
                nccopydimension(ncin,ncout,edimension,copyvalues=True)
        #ncout.createVariable(variable,ncin.variables[variable].typecode(),ncin.variables[variable].dimensions)
        # workaround: functions are now considered an output to be float64: better make the function make arrays
        # that have the same output type!! create variable according to the output of dataouttemp!!
        ncout.createVariable(varoutdef,ncin.variables[varin].typecode(),ncin.variables[varin].dimensions)
        if copyattr:
            nccopyattrvar(ncin,ncout,varin=varin,varout=varoutdef)
            # for attr in dir(ncin.variables[varin]):
            #     if attr not in ['assignValue', 'getValue', 'typecode']:
            #         setattr(ncout.variables[varoutdef],attr,getattr(ncin.variables[varin],attr))
    if copyvalues == True:
        ncout.variables[varoutdef][:] = ncin.variables[varin][:]

