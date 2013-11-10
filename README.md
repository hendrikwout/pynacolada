===========
PynaColaDa
===========


Software package for easy and customized Processing huge amounts of gridded Climate Data.

Just imagine that your tons of terrabytes of data can be processed  in (parametric) space or
time (or combination) with only a few (readable!) lines of code waiting only maybe just one
night.  For this task, we would obviously fall back to tools like cdo or ncl.  However, we
are stuck to the data analysis options that these provide. That's where pynacolada comes
into play.  We can now apply any arbitrary pre-defined or user-defined function analysis on a
huge dataset, and with great performance (from the numpy/matlib library), and very simply.


The power of tools lies in their examples. Typical usage of the package, in particular ncmultifunc,
looks like this::

    #!/usr/bin/env python
    # 2013-11-08: Below, you find  just a code snippet from the first tests.. Examples will be added very soon
   #print fnout
            
    # Example 1: calculate the mean scalar wind speed of the first 10 layers
    import pynacolada as pcl
    from Scientific.IO import NetCDF
    import os
    import numpy as np
    import pylab as pl
    
    fnin = 'laf2009010100.nc'
    #print fnin
    # fobjin = open(fnin,'rb')
    fin = NetCDF.NetCDFFile(fnin,'r')
    fnout = 'laf2009010100_out.nc'
    os.system('rm '+fnout)
    #print fnout
    # fobjout = open(fnout,'wb+''rlat')
    fout = NetCDF.NetCDFFile(fnout,'w')
    # input data definitions
    datin =  [{'file': fin, \
               'varname': 'U', \
               'dsel': {'level' : range(30,40,1)}, \
               'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
              },\
              {'file': fin, \
               'varname':'V', \
               'dsel': {'level' : range(30,40,1)},
               'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
               }\
             ]
    # output data definitions
    datout = [{'file': fout, \
               'varname': 'u'}]
    # function definition:
    func = lambda U,V: np.array([np.mean(np.sqrt(U**2+V**2),axis=0 )])
    dnamsel = ['level',]
    pcl.pcl(func,dnamsel,datin,datout,appenddim=True)
    print 'output data written to:',fnout

A Section
=========




