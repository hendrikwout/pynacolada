from numpy import *
from datetime import *
from matplotlib.dates import *

# get corresponding format for netcdf file
def ncgetformat(fmt):    
    if fmt == float64: #double
        return 'd'
    elif fmt == float32:
        return 'f'
    elif fmt == int:
        return 'i'
    elif fmt == long:
        return 'l'
    elif fmt == datetime:
        return 'd' 
    else:
        print (' WARNING: format '+fmt+' could not be discovered! Assuming that it is double.')
        return 'd'

def csv2netcdf(infile,outfile,sep=None,nalist = [''],formatlist=[],refdat = datetime(2000,1,1),tunits = 'hours',customnames=[]):
    '''
    purpose: convert a csv-file to a netcdf timeseries. 

    infile: reference to a filestream containing csv data (example: open('file.csv'))
    outfile: reference to a writable netcdf-file NetCDF.NetCDFFile('file.nc')
    nalist: list with characters that should be skipped. It will be assigned as nan
    formatlist: a list of pairs, in which one specifies the format for each csv-column. The first element
    of a pair is the data type, the second element is the data format (e.g. like '%Y/%m/%d for a date)?.
    Example: ((datetime,'%Y/%m/%d',(double,nan))
    tunits: the unit for the 'time' coordinate (default: 'hours')
    refdat: reference date. (will be used to make the time dimension e.g. 'hours' since refdat)
    ''' 

    # todo: 
    tell = infile.tell()
    leninfile = len(infile.readlines()) - 1
    infile.seek(tell)
    colnames = infile.readline().replace('\n','').split(sep)

    for customname in customnames:
        colnames[customname[0]] = customname[1]

    # # workaround ncview crash for underscores
    # for colname in colnames:
    #     colname.replace('_','')
    #formatlist.extend(repeat((nan,nan),len(colnames) - len(formatlist)))
    for i in range(len(colnames) - len(formatlist)):
        formatlist.append((double,nan))

    outfile.createDimension('time',leninfile)
    outfile.createVariable('time','d',('time',))
    outfile.createVariable('datetime','d',('time',))
    if (tunits not in nalist):
        setattr(outfile.variables['time'],\
                'units', \
                tunits +' since '+ datetime.strftime(refdat,'%Y-%m-%d %H:%M:%S'))

    for icn,ecn in enumerate(colnames):
        outfile.createVariable(ecn,ncgetformat(formatlist[icn][0]),('time',))
    for iline,eline in enumerate(infile):
        currline = eline.replace('\n','').split(sep)
        colnum = min(len(currline),len(colnames))
        for icn in range(colnum):
            ecn = colnames[icn]
            if currline[icn] not in nalist:
                if formatlist[icn][0] == datetime:
                    outfile.variables[ecn][iline] = date2num(datetime.strptime(currline[icn],formatlist[icn][1]))
                else:
                    try:
                        outfile.variables[ecn][iline] = formatlist[icn][0](currline[icn])
                    except:
                        outfile.variables[ecn][iline] = nan
            else:
                outfile.variables[ecn][iline] = nan

        uzformatlist = zip(*formatlist)
        # get date of string
        dt = refdat
        dtype = datetime
        if dtype in uzformatlist[0]:
            icn = uzformatlist[0].index(dtype)
            if currline[icn] not in nalist:
                dt = datetime.strptime(currline[icn],uzformatlist[1][icn])
        dtype = date
        if dtype in uzformatlist[0]:
            icn = uzformatlist[0].index(dtype)
            if currline[icn] not in nalist:
                dt = datetime.strptime(currline[icn],uzformatlist[1][icn])

        # tbi loop over every 'daytime' type and add them together
        dtype = 'daytime'
        if dtype in uzformatlist[0]:
            icn = uzformatlist[0].index(dtype)
            if currline[icn] not in nalist:
                currval = double(currline[icn])
                sec = 0.
                if uzformatlist[icn][1] == 'HMint':
                    sec = (int(currval/100.)*60. + (currval - int(currval/100.)*100.))*60.
                dt = dt + timedelta(seconds = sec)

        # output to netcdf
        outfile.variables['datetime'][iline] = date2num(dt)

        if tunits == 'hours':
            outfile.variables['time'][iline] = \
                    (dt-refdat).total_seconds()/3600.
        if tunits == 'days':
            outfile.variables['time'][iline] = \
                    (dt-refdat).total_seconds()/3600./24.
        if tunits == 'seconds':
            outfile.variables['time'][iline] = \
                    (dt-refdat).total_seconds()
        if tunits == 'minutes':
            outfile.variables['time'][iline] = \
                    (dt-refdat).total_seconds()/60.

def ncwritedatetime(ncfile,dt,tunits = None, refdat = None):
    # purpose: write given time coordinate to ncdf file
    # dt: array of datetimes
    # tunits: time unit (can be either 'hour', 'days', 'minutes','seconds',
    # refdat: reference date from which the timesteps are represented

    if refdat == None: refdat = dt[0]
    if 'time' not in ncfile.dimensions:
        ncfile.createDimension('time',len(dt))
    if 'time' not in ncfile.variables:
        ncfile.createVariable('time','d',('time',))

    if tunits == None:
        try:
            if (dt[1] - dt[0]) == timedelta(years=1): tunits = 'years'
            elif (dt[1] - dt[0]) == timedelta(days=1): tunits = 'days'
            elif (dt[1] - dt[0]) == timedelta(hours=1): tunits = 'hours'
            elif (dt[1] - dt[0]) == timedelta(minutes=1): tunits = 'minutes'
            elif (dt[1] - dt[0]) == timedelta(hours=1): tunits = 'seconds'
            else: tunits = 'hours'
        except:
            tunits = 'hours'

    setattr(ncfile.variables['time'],\
            'units', \
             tunits +' since '+ datetime.strftime(refdat,'%Y-%m-%d %H:%M:%S'))


    for idt,edt in enumerate(dt):
        if tunits == 'hours':
            ncfile.variables['time'][idt] = \
                    (edt-refdat).total_seconds()/3600.
        if tunits == 'days':
            ncfile.variables['time'][idt] = \
                    (edt-refdat).total_seconds()/3600./24.
        if tunits == 'seconds':
            ncfile.variables['time'][idt] = \
                    (edt-refdat).total_seconds()
        if tunits == 'minutes':
            ncfile.variables['time'][idt] = \
                    (edt-refdat).total_seconds()/60.

