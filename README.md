# pynacolada
Library for easily processing big netCDF files and xarrays, and manage them in
hierarchical processing flows.

# Description
Ever want to manipulate databases of huge xarrays in an easy,
efficient, memory-friendly, and traceable way? You may have noticed memory
limitations and loss of overview of all the sorts of the datasets.  This
library aims to tackle these challenges.

On the one hand, this package implements an apply_func routine that circumvents
the memory issue by automatically dividing the xarray into chunks that fit your
computer memory. On the fly writing to disk, parallel processing of
(overlapping) chunks are supported simultaneously.

On the other hand, the library allows anyone to create modular data processing flows
for processing different datasets with different formats, in which the multiple
processes are executed in a hierarchical order and in parallel where possible.
Herewidth, It implements a data management framework in which process arguments
on the one hand and xarray (netcdf) attributes on the other hand are propagated
in between the processing modules.  Therefore, attributes from the
xarray/netcdf files in the databases are tracked with pandas tables.While
executing the multiple processing steps, the Pynacolada library checks whether
particular data is already (being) generated.  In that case, another build of
the data is omitted, and only the reference (pandas table) to the existing data
is provided instead.

The library is currently used for different projects climate services developed
by the Climate team of the Flemish Institute for Technological Research (VITO)
in Belgium, amongst others, climtag.vito.be. For examples and usage, please
contact hendrik.wouters@vito.be.


