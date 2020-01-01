# pynacolada
Library for easily IO-processing with huge netCDF xarrays

# purpose
Ever want to manipulate huge xarrays? You may have noticed memory limitations. This package implements an apply_func routine that circumvents the memory issue by automatically dividing the xarray into chunks that fit your computer memory. On the fly writing to disk is also supported.

