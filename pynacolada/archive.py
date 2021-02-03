import glob
import os
import pandas as pd
import xarray as xr
import numpy as np
import numpy as np
import datetime as dt
import itertools
import yaml
import sys
from tqdm import tqdm
import tempfile
from . import apply_func

def parse_to_dataframe(list_or_dict_or_dataframe):
    if type(list_or_dict_or_dataframe) == pd.core.frame.DataFrame:
        return list_or_dict_or_dataframe
    elif type(list_or_dict_or_dataframe) == list:
        return pd.DataFrame(list_or_dict_or_dataframe)
    elif type(list_or_dict_or_dataframe) == dict:
        return pd.DataFrame.from_dict([dict(zip(list_or_dict_or_dataframe.keys(),v)) for v in itertools.product(*list_or_dict_or_dataframe.values())])


# test ;lkjdfg

def empty_multiindex(names):
    """
    Creates empty MultiIndex from a list of level names.
    """
    return pd.MultiIndex.from_tuples(tuples=[(None,) * len(names)], names=names)

class archive (object):
    def __init__(self,path=None,*args,**kwargs):
        self.lib_dataarrays = pd.DataFrame(index=empty_multiindex(['variable','source','time','space']),columns = ['path','absolute_path']).iloc[1:]
        self.dataarrays = {}
        self.coordinates = {}

        self.settings_keys = ['file_pattern','mode']
        print('Creating generic functions to set attributes')
        for key in self.settings_keys:
            print('creating function self.set_'+key)
            self.__dict__['set_'+key] = lambda value: self.__setattr__(key,value)
        print('Loading default settings')
        self.file_pattern = '"variable"_"source"_"time"_"space".nc'
        self.mode = 'active'

        self.path_pickle = None
        print('Loading datasets')
        if path is not None:
            self.load(path,*args,**kwargs)

    def copy(self):
        return self.archive.apply(lambda x: x.copy())

    def sel(self,sel):
        return self.archive.apply(lambda x: x.sel(sel))

    def sel_lib(self,sel):
        lib_dataarrays_out = self.lib_dataarrays_out[sel]
        archive_out = archive()
        for index,lib_dataarray in lib_dataarays_out.iterrows():
            archive_out.add_dataarray(self.dataarrays[index])

    def remove(self,index,delete_on_disk=False):
        self.dataarrays[index].close()
        del self.dataarrays[index]
        if delete_on_disk:
            os.system('rm '+self.lib_dataarrays.loc[index].absolute_path)
        print(self.lib_dataarrays.loc[index].absolute_path_as_cache)
        if (self.lib_dataarrays.loc[index].absolute_path_as_cache is not None):
            print(np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache))
            print(np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache) == True)
        if (self.lib_dataarrays.loc[index].absolute_path_as_cache is not None) and (np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache == False)) :
            CMD = 'rm '+self.lib_dataarrays.loc[index].absolute_path_as_cache
            print('removing cached file:',CMD)
            
        self.lib_dataarrays.drop(index=index,inplace=True)

    def close(self):
        lib_dataarrays_temp = self.lib_dataarrays.copy()
        for index,columns in lib_dataarrays_temp.iterrows():
            self.remove(index=index)

        del lib_dataarrays_temp

    def add_from_dataset(self,Dataset_or_filepath,variables=None,**kwargs):
        if type(Dataset_or_filepath).__name__ == 'str':
            Dataset = xr.open_dataset(Dataset_or_filepath)
            kwargs['absolute_path'] = os.path.abspath(Dataset_or_filepath)
            kwargs['absolute_for_reading'] = kwargs['absolute_path']
        else:
            Dataset = Dataset_or_filepath

        if variables is None:
            variables= []#Dataset.variables
            for variable in Dataset.variables:
                print(Dataset.dims)
                if variable not in Dataset.dims:
                    variables.append(variable)
        for variable in variables:
            self.add_dataarray(Dataset[variable],**kwargs)

        Dataset.close()

    def add_dataarray(self,DataArray_or_filepath,skip_unavailable= False,release_dataarray_pointer=False,cache_to_tempdir=False,cache_to_ram=False,**kwargs):
        #DataArray = None
        if type(DataArray_or_filepath).__name__ == 'str':
            filepath = DataArray_or_filepath
            if (cache_to_tempdir is None) or cache_to_tempdir:
                if type(cache_to_tempdir) is not str:
                    cache_to_tempdir = tempfile.gettempdir()
                #filepath_as_cache = cache_to_tempdir+'/'+os.path.basename(filepath)

                filepath_as_cache = tempfile.mktemp(prefix=os.path.basename(filepath)[:-3]+'_',suffix='.nc',dir=cache_to_tempdir)
                CMD='cp '+filepath+' '+filepath_as_cache

                print('caching to temporary file: ',CMD)
                os.system(CMD)
                filepath_for_reading = filepath_as_cache
            else:
                filepath_as_cache = None
                filepath_for_reading = filepath
                if cache_to_ram:
                    CMD='cat '+filepath_for_reading+' > /dev/null'
                    print('caching to ram:',CMD)
                    os.system(CMD)

            # ncvariable: variable as seen on disk
            # variable (= DataArray.name): variable as considered in the library
            ncvariable = None # netcdf variable as seen on disk, not necessarily in the library
            if ('ncvariable' not in kwargs.keys()) or ((type(kwargs['ncvariable']).__name__ == 'float') and np.isnan(kwargs['ncvariable'])):
                if 'variable' in kwargs.keys():
                    ncvariable = kwargs['variable']

                print('Opening file:',filepath_for_reading, '(original file: '+filepath+')')
                if ncvariable is not None:
                    # print('reading',filepath,ncvariable)
                    # try:
                    #     print('trying with open_dataarray first because of obscure performance decreases with xr.open_dataset')
                    #     DataArray = xr.open_dataarray(filepath)
                    #     if DataArray.name != kwargs['ncvariable']:
                    #         DataArray.close()
                    #         del DataArray
                    #         print('first variable is not the correct ncvariable. Trying it with xr.open_dataset...')
                    #         raise ValueError('first variable is not the correct ncvariable')
                    # except:
                    try:
                        
                        ds = xr.open_dataset(filepath_for_reading)
                        DataArray = ds[ncvariable]
                        ds.close()
                        del ds
                        #kwargs['ncvariable'] = ncvariable
                    except:
                        DataArray = xr.open_dataarray(filepath_for_reading)
                else:
                    DataArray = xr.open_dataarray(filepath_for_reading)
                kwargs['ncvariable'] = DataArray.name 
            else: 
                ds = xr.open_dataset(filepath_for_reading)
                DataArray = ds[kwargs['ncvariable']]
                ds.close()
                del ds
            # except:
            #     print ( "Warning! Error while reading from disk: ", sys.exc_info()[0])
            #     if not skip_unavailable:
            #         raise ('...stopping' )
            #     else:
            #         print('...skipping' )

            kwargs['absolute_path'] = os.path.abspath(filepath)
            kwargs['absolute_path_for_reading'] = os.path.abspath(filepath_for_reading)
            kwargs['absolute_path_as_cache'] = (None if filepath_as_cache is None else os.path.abspath(filepath_as_cache))
        else:
            DataArray = DataArray_or_filepath

            #kwargs['absolute_path'] = None 

        # for key in self.lib_dataarrays.index.names:
        #     if key not in (list(DataArray.attrs.keys())+['variable']):
        #         raise ValueError(key+' needs to be in DataArray.attrs')

        # if DataArray is None:
        #     print('Skipping ',DataArray_or_filepath,kwargs)
        # else:
        if not DataArray is None:

            dict_index = {}
            dict_columns = {}
            
            if DataArray.name is None:
                raise ValueError ('input dataarray should have a name')
            
            if 'variable' not in kwargs.keys():
                kwargs['variable'] = DataArray.name

            DataArray.name = kwargs['variable']

            dict_index['variable'] = DataArray.name
            
            for key,value in DataArray.attrs.items():
               if key in self.lib_dataarrays.index.names:
                   dict_index[key] = value
               else:
                   dict_columns[key] = value

            # for key in self.lib_dataarrays.index.names:
            #     if key in kwargs.keys():
            #         dict_index[key] = kwargs[key]
            for key,value in kwargs.items():
                if key in self.lib_dataarrays.index.names:
                    dict_index[key] = kwargs[key]
                else:
                    dict_columns[key] = kwargs[key]
            if ('time' not in dict_index.keys()) or (dict_index['time'] is None) or (type(dict_index['time']).__name__ == 'float') and (  np.isnan(dict_index['time']).any()):
                print('Guessing time coordinate from DataArray')
                # is month type
                
                #monthly spacing
                if np.apply_along_axis(lambda y: np.sum((y[1:] - y[:-1] != 1),0),0,np.vectorize(lambda x: int(x[:4])*12+int(x[5:7]))(DataArray.time.values.astype('str'))).item() == 0:
                    dict_index['time'] = \
                            'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
                # also monthly 
                elif (not np.any( ~(np.vectorize(lambda x: x[8:])(DataArray.time.values.astype('str')) == '01T00:00:00.000000000'))):
                    dict_index['time'] = \
                            'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
                elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != np.array(86400000000000, dtype='timedelta64[ns]')):
                    #daily
                    dict_index['time'] = 'daily_'+np.datetime_as_string(DataArray.time[0].values,unit='D')+'_'+np.datetime_as_string(DataArray.time[-1].values,unit='D')
                elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != dt.timedelta(days=1)):
                    dict_index['time'] = \
                            'daily_'+str(DataArray.time[0].values)[:10]+'_'+str(DataArray.time[-1].values)[:10]
                else:
                    raise ValueError('time dimension not implemented')

                #DataArray.attrs['time'] = dict_index['time']

            renamings = {'lat':'latitude','lon':'longitude'}
            for key,value in renamings.items():
                if key in DataArray.dims:
                    DataArray = DataArray.rename({key:value})

            # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
            space_coordinates = list(DataArray.dims)
            for key in self.lib_dataarrays.index.names:
                if key in space_coordinates:
                    space_coordinates.remove(key)

            if ('space' not in dict_index.keys()) or (dict_index['space'] is None):
                spacing = {}
                for coordinate in space_coordinates:
                    spacing_temp = (DataArray[coordinate].values[1] - DataArray[coordinate].values[0])
                    if not np.any(DataArray[coordinate][1:].values != (DataArray[coordinate].values[:-1] + spacing_temp)):
                        spacing[coordinate] = spacing_temp
                    else:
                        spacing[coordinate] = 'irregular'
                dict_index_space = [key+'_'+str(value) for key,value in spacing.items()]
                dict_index_space ='_'.join(dict_index_space) 
                dict_index['space'] = dict_index_space
                #DataArray.attrs['space'] = dict_index_space


            for key,index in dict_index.items():
                if key not in self.coordinates:
                    if key not in ['variable','source','space',]:
                            self.coordinates[key] = DataArray[key]
                    if key == 'space':
                        self.coordinates[key] = []
                        for coordinate in space_coordinates:
                            self.coordinates[key].append(DataArray[coordinate])

            for key in self.lib_dataarrays.index.names:
                if (key not in dict_index.keys()) or (dict_index[key] is None):
                    raise ValueError ('Could not track key "'+key+'" that is required for the archive index.')
            index = tuple([dict_index[key] for key in self.lib_dataarrays.index.names]) 


            self.dataarrays[index] = DataArray
            DataArray.close()
            del DataArray

            if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
               self.dataarrays[index].close()

            for key,value in dict_index.items():
                if key == 'variable':
                    self.dataarrays[index].name = value
                else:
                    self.dataarrays[index].attrs[key] = value

            if 'absolute_path' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path'] = None
            if 'absolute_path_for_reading' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path_for_reading'] = None
            if 'absolute_path_as_cache' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path_as_cache'] = None
            if 'path' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['path'] = None

            self.lib_dataarrays.loc[index] = None
            for key,value in dict_columns.items():
                if key not in self.lib_dataarrays.columns:
                    self.lib_dataarrays[key] = ''
                if index not in self.lib_dataarrays.index:
                    self.lib_dataarrays.loc[index] = ''
                self.lib_dataarrays[key].loc[index] = value
                if key not in [ 'absolute_path_as_cache','absolute_path_for_reading','absolute_path','path']:
                    self.dataarrays[index].attrs[key] = value

            self.lib_dataarrays.sort_index(inplace=True)
            if release_dataarray_pointer:
                print('closing',index)
                self.dataarrays[index].close()
                if cache_to_tempdir:
                    CMD='rm '+filepath_as_cache
                    print('Released pointer, so removing cached file: ',CMD)
                    os.system(CMD)
                #del self.dataarrays[index]

    #         if self.path_pickle:# and (type(self.lib_dataarrays.loc[index].absolute_path) != str):
    #             self.dump()


            # if attrs is not None:
            #     for key,value in attrs.items():
            #         DataArray

            # for key,value in kwargs.items():
            #     if key not in self.lib_dataarrays.columns:
            #         self.lib_dataarrays[key] = ''
            #     self.lib_dataarrays.loc[index][key] = value

    #    def intersect_times(self,DataArray_input):
    def dataarrays_apply(self,function,query=None,inplace=False, attrs=None):
        # if query is not None:
        #     lib_dataarrays_out = self.lib_dataarrays.query(query,engine='python').copy()
        # elsepath
        #     lib_dataarrays_out = self.lib_dataarrays.copy()

        archive_out = archive()
        for index,columns in self.lib_dataarrays.iterrows():
            dataarray_out_temp = function(self.dataarrays[index])
            for key,value in self.dataarrays[index].attrs.items():
                dataarray_out_temp.attrs[key] = value
            if attrs is not None:
                for key,value in attrs.items():
                    dataarray_out_temp.attrs[key] = value
            archive_out.add_dataarray(dataarray_out_temp)
            
        return archive_out

    def apply_virtual(self,func,groupby=None,apply_merge=[],apply_merge_out=[],archive_out = None, inherit_attributes = True,extra_attributes={}, **kwargs):

        if (archive_out is None) and (self.path_pickle is None):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')

        temp_df = self.lib_dataarrays.reset_index() 


        if type(apply_merge) == pd.core.frame.DataFrame:
            apply_merge_df = apply_merge
        elif type(apply_merge) == list:
            apply_merge_df = pd.DataFrame(apply_merge)
        elif type(apply_merge) == dict:
            apply_merge_df = pd.DataFrame.from_dict([dict(zip(apply_merge.keys(),v)) for v in itertools.product(*apply_merge.values())])
        apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        if type(apply_merge_out) == pd.core.frame.DataFrame:
            apply_merge_out_df = apply_merge_out
        elif type(apply_merge_out) == list:
            apply_merge_out_df = pd.DataFrame(apply_merge_out)
        elif type(apply_merge_out) == dict:
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out.keys(),v)) for v in itertools.product(*apply_merge_out.values())])
        
        if len(apply_merge_out_df) == 0:
            print('creating automatic single output table')
            apply_merge_out_dict = {}
            for column in apply_merge_df.columns:
                apply_merge_out_dict[column] = ['from__'+'__'.join(apply_merge_df[column].unique())]
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out_dict.keys(),v)) for v in itertools.product(*apply_merge_out_dict.values())])
                

        apply_merge_out_index = pd.MultiIndex.from_frame(apply_merge_out_df)

        

        #apply_merge_pandas = pd.DataFrame([{'variable':'msshf_0001'},{'variable':'mslhf_0001','source':'cds_era5'}])
        
        #self.lib_dataarrays.index.names
        if groupby is None:
            groupby = list(self.lib_dataarrays.index.names)
            for key in apply_merge_df.columns:
                groupby = list(filter((key).__ne__, groupby))

        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        if write_mode == 'create_new_archive':
            archive_out = archive()
            archive_out.dump(path_archive_out)
        elif write_mode == 'add_to_current_archive':
            archive_out = self

        xr_outputs = {}
        for index,group in temp_df.groupby(groupby):
            if type(index) is not tuple:
                index_multi = (index,)
            else:
                index_multi = index

            group_columns = group.set_index(list(apply_merge_df.columns))
            if type(group_columns.index) == pd.core.indexes.base.Index:
                MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
                group_columns.index = MultiIndex_from_Single_Index(group_columns.index)

            all_variables_available_in_this_group = True
            if len(apply_merge_index.intersection(group_columns.index)) != len(apply_merge_index):
             all_variables_available_in_this_group = False
            # if not group_columns.loc[apply_merge_index][groupby].isnull().any().any():
            #  all_variables_available_in_this_group = False

            if all_variables_available_in_this_group:
                  dataarrays_for_func = []

                  for index_group,columns in group_columns.loc[apply_merge_index].iterrows():
                      index_array_dict =   {**dict(columns),**dict(zip(apply_merge_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
                      index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])

                      if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
                          print('to be implemented')
                          import pdb; pdb.set_trace()
                      else:
                          dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])

                  filenames_out = []
                  
                  attributes = []
                  ifile = 0
                  for index_group,group_columns in apply_merge_out_df.iterrows():
                      index_array_out_dict = {**dict(zip(groupby,index_multi)),**dict(zip(apply_merge_out_df.columns,group_columns))}
                      attributes.append(index_array_out_dict)
                      index_array_out_tuple_ordered =  tuple([index_array_out_dict[key] for key in self.lib_dataarrays.index.names])

                      if inherit_attributes:
                          for key,value in dataarrays_for_func[min(len(dataarrays_for_func),ifile)].attrs.items():
                              if (key not in self.lib_dataarrays.index.names)  and \
                                 ( inherit_attributes or ((type(inherit_attributes) is list) and (key in inherit_attributes))) and \
                                 (key not in attributes[-1].keys()):
                                  attributes[-1][key] = value

                      for key,value in extra_attributes.items():
                          attributes[-1][key] = value

                      # if (archive_out.file_pattern is None):
                      #     raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern") 
                      #filenames_out.append(os.path.dirname(archive_out.path_pickle)+'/'+''.join(np.array(list(zip(archive_out.file_pattern.split('"')[::2],[attributes[-1][key] for key in archive_out.file_pattern.split('"')[1::2]]+['']))).ravel()))
                      ifile +=1
                  #  ifile = 0
                  #  for index_group,group_columns in apply_merge_out_df.iterrows():
                  #      index_array_out_tuple_ordered =  tuple([attributes[ifile][key] for key in archive_out.lib_dataarrays.index.names])
                  #      if index_array_out_tuple_ordered in archive_out.dataarrays.keys():
                  #          print('forcing to overwrite data for ',index_array_out_tuple_ordered,)
                  #          self.remove(index_array_out_tuple_ordered,delete_on_disk=True)
                  #      ifile +=1
                  

                  # for filename_out in filenames_out:
                  #     os.system('mkdir -p '+os.path.dirname(filename_out))

                  temp_dataarrays = func(tuple(dataarrays_for_func))

                    # if type(temp_dataarrays) != tuple:
                    #     print('this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                    #     idataarray = 0
                    #     for key,value in attributes[idataarray].items():
                    #         if key not in ['variable','absolute_path','path']:
                    #             temp_dataarrays.attrs[key] = value
                    #         if key == 'variable':
                    #             temp_dataarrays.name = value
                    #     #import pdb;pdb.set_trace()
                    #     temp_dataarrays.to_netcdf(filenames_out[idataarray])
                    #     temp_dataarrays.close()
                    # else:
                    #     for idataarray in range(len(temp_dataarrays)):
                    #         for key,value in attributes[idataarray].items():
                    #             if key not in ['variable','absolute_path','path']:
                    #                 temp_dataarrays[idataarray].attrs[key] = value
                    #             if key == 'variable':
                    #                 temp_dataarrays[idataarray].name = value

                    #     for idataarray in range(len(temp_dataarrays)):
                    #         temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                    #         temp_dataarrays[idataarray].close()

                  for idataarray,dataarray in enumerate(temp_dataarrays):
                      for key,value in attributes[idataarray].items():
                          dataarray.attrs[key] = value
                      archive_out.add_dataarray(dataarray)
        if write_mode == 'create_new_archive':
            return archive_out



#             if type(group_columns_out.index) == pd.core.indexes.base.Index:
#                 MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
#                 group_columns_out.index = MultiIndex_from_Single_Index(group_columns_out.index)
#            
#             # not_all_arrays_available_in_this_group = group_columns.loc[apply_merge_out_index][groupby].isnull().any().any()
#             #       dataarrays_for_func = []
# 
#             import pdb;pdb.set_trace()
#             for index_group,columns in group_columns_out.loc[apply_merge_out_index].iterrows():
#                 index_array_dict =   {**dict(columns),**dict(zip(apply_merge_out_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
#                 index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])
#                 # dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])
# 




                #group_columns.loc[apply_merge_index]
            

        # temp_df.groupby(self.lib_dataarrays.index.names
        # if kwargs.keys() != 1:
        #     raise ValueError('length different from 1 is not allowed')

    def apply_virtual2(self,func,divide_into_groups_extra=None, apply_groups_in=[],apply_groups_out=[], archive_out=None,inherit_attributes=True,extra_attributes={},**kwargs ):


        divide_into_groups = []
        for name in self.lib_dataarrays.index.names:
            if ((name not in apply_groups.keys()) and (name not in apply_groups.keys())):
                divide_into_groups.append(name)
        for name in divide_into_groups_extra:
            if name not in divide_into_groups:
                divide_into_groups.append(name)
        
        for index,group in self.lib_dataarrays.reset_index().groupby(divide_into_groups):

            def allways_multi_index (idx):
                if type(idx) is not tuple:
                    return (idx,)
                else:
                    return idx

            multi_index = allways_multi_index(index)

            apply_groups_columns = []
            for key in apply_groups_in.keys():
                apply_groups_columns.append(key)
            for key in apply_groups_out.keys():
                if key not in apply_groups_columns:
                    apply_groups_columns.append(key)

            apply_this_group_in = pd.dataframe(columns=apply_groups_columns)

                 





        def parse_to_dataframe(list_or_dict_or_dataframe):
            if type(list_or_dict_or_dataframe) == pd.core.frame.DataFrame:
                return list_or_dict_or_dataframe
            elif type(list_or_dict_or_dataframe) == list:
                return pd.DataFrame(list_or_dict_or_dataframe)
            elif type(list_or_dict_or_dataframe) == dict:
                return pd.DataFrame.from_dict([dict(zip(list_or_dict_or_dataframe.keys(),v)) for v in itertools.product(*list_or_dict_or_dataframe.values())])

        apply_groups_in_df = parse_to_dataframe(apply_groups_in)
        apply_groups_out_df = parse_to_dataframe(apply_groups_out)

        # import pdb;pdb.set_trace()


        # apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        # if type(apply_groups_out) == pd.core.frame.DataFrame:
        #     apply_groups_out_df = apply_groups_out
        # elif type(apply_groups_out) == list:
        #     apply_groups_out_df = pd.DataFrame(apply_groups_out)
        # elif type(apply_groups_out) == dict:
     




        # apply_groups_in_out_pre = 

        # if archive_out is not None:
        #     write_mode = 'add_to_external_archive'
        # else:
        #     write_mode = 'add_to_current_archive'



        #     apply_groups_index




    #def cdo(self,cdostring,):
    def apply_func(self,
            func, 
            xarray_function_wrapper=apply_func,
            dataarrays_wrapper = lambda *x: (*x,),
            groupby=None,
            apply_groups_in=[],
            apply_groups_out=[],
            divide_into_groups_extra = [],
            archive_out = None,
            mode = 'numpy_output_to_disk_in_chunks', 
            inherit_attributes = False,
            query=None,
            extra_attributes={}, 
            post_apply=None,
            initialize_array=None,
            copy_coordinates=False,
            **kwargs):
        #apply_groups_in = {'variable':['aridity'],'source':[None]}
        #apply_groups_out={'variable':['aridity'],'source':[lambda labels: labels[0].replace('historical','rcp45'),lambda labels: labels[0].replace('historical','rcp85')]}
        #archive_out = pcd.archive()
        #mode='xarray'
        
        
        if (archive_out is None) and (self.path_pickle is None) and (mode != 'xarray'):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')
        
        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        if write_mode == 'create_new_archive':
            archive_out = archive()
            archive_out.dump(path_archive_out)
        elif write_mode == 'add_to_current_archive':
            archive_out = self
        

        
        
        apply_groups_in_df = parse_to_dataframe(apply_groups_in)
        apply_groups_out_df = parse_to_dataframe(apply_groups_out)
        
        divide_into_groups = []
        for name in self.lib_dataarrays.index.names:
            if (name not in apply_groups_in_df.columns):
                divide_into_groups.append(name)
        for name in divide_into_groups_extra:
            if name not in divide_into_groups:
                divide_into_groups.append(name)
        
        if query is not None:
            if type(query) == str:

                read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
            elif type(query) ==  pd.DataFrame:
                read_lib_dataarrays = query
            else:
                raise ValueError('type of input query '+query+'not implemented')
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()
        groups_in_loop = read_lib_dataarrays.reset_index().groupby(divide_into_groups)
        print('Looping over data array input groups: ',list(groups_in_loop))
        for idx,group in tqdm(groups_in_loop):
        
            def always_tuple (idx):
                if type(idx) is not tuple:
                    return (idx,)
                else:
                    return  idx
            def always_multi_index (index):
            
                if type(index) == pd.core.indexes.base.Index:
                     return pd.MultiIndex.from_tuples([x.split()[::-1] for x in index],names=(index.name,))
                else:
                    return index
            
            apply_this_group_in_df = apply_groups_in_df.copy()
            for idx_group_in,row in apply_this_group_in_df.iterrows():
                for column in apply_this_group_in_df.columns:
                    if row[column] is None:
                        #import pdb;pdb.set_trace()
                        apply_this_group_in_df[column].loc[idx_group_in] = group.loc[:,column].unique()
            
            for column in apply_this_group_in_df.columns:
                apply_this_group_in_df = apply_this_group_in_df.explode(column)
            
            multi_idx = always_tuple(idx)
            
            check_group_columns = []
            for column in apply_this_group_in_df.columns: 
                if (~apply_this_group_in_df[column].isnull()).any(): 
                    check_group_columns.append(column) 
            
            group_reduced = group[check_group_columns].drop_duplicates().reset_index().drop(columns='index')
            apply_this_group_in_df_reduced = apply_this_group_in_df[check_group_columns].drop_duplicates().reset_index().drop(columns='index')
            
            if len(apply_groups_in_df) == 0:
                all_input_available_in_this_group = True
            else:
                apply_this_group_in_reduced_index = pd.MultiIndex.from_frame(apply_this_group_in_df_reduced)
                group_reduced_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df_reduced.columns))
                group_reduced_indexed_for_apply_groups_in.index = always_multi_index(group_reduced_indexed_for_apply_groups_in.index)
                
                all_input_available_in_this_group = not (len(apply_this_group_in_reduced_index) != len(apply_this_group_in_reduced_index.intersection(group_reduced_indexed_for_apply_groups_in.index)))
            
            #group_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df.columns))
            
            
            
            
            
            # 
            # 
            # 
            # apply_this_group_in_index = pd.MultiIndex.from_frame(apply_this_group_in_df)
            # group_indexed_for_apply_groups_in.index = always_multi_index(apply_this_group_in_df.index)
            # 
            # stop
            # 
            # #group_selection_index = pd.MultiIndex.from_frame(frame_for_selecting_xarrays_from_current_group)
            # 
            # all_input_available_in_this_group = not (group_indexed_for_apply_groups_in.index != apply_groups_in_index).any()
            # 
            if all_input_available_in_this_group:
            
                if len(apply_this_group_in_df) == 0:
                    table_this_group_in = group
                else:
                    apply_this_group_in_index = pd.MultiIndex.from_frame(apply_this_group_in_df)
                    group_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df.columns))
                    group_indexed_for_apply_groups_in.index = always_multi_index(group_indexed_for_apply_groups_in.index)
                
                    # apply_this_group_in_df = apply_groups_in_df.copy()
                    # for idx_group_in,row in apply_this_group_in_df.iterrows():
                    #     for column in apply_this_group_in_df.columns:
                    #         if row[column] is None:
                    #             apply_this_group_in_df[column].loc[idx_group_in] = group.loc[:,column].unique()
                
                    
                    # import pdb; pdb.set_trace()
                    # apply_this_group_in_index = apply_this_group_in_index.intersection(group_indexed_for_apply_groups_in.index,sort=None)
                    # import pdb; pdb.set_trace()
                    table_this_group_in = group_indexed_for_apply_groups_in.loc[apply_this_group_in_index]

                
                #apply_groups_out_index = pd.MultiIndex.from_frame(apply_groups_out_df)
                    # group_indexed_for_apply_groups_out = group.set_index(list(apply_groups_out_df.columns))
                    # group_indexed_for_apply_groups_out.index = always_multi_index(group_indexed_for_apply_groups_out.index)
                
            
                apply_groups_out_df_this_group = apply_groups_out_df.copy()
                

                print('converting label functions where necessary')
                for idx_group_out,row in apply_groups_out_df.iterrows():
                    for key,value in row.items():
                        if type(apply_groups_out_df_this_group.loc[idx_group_out,key]).__name__ == 'function':
                            apply_groups_out_df_this_group.loc[idx_group_out,key]= apply_groups_out_df.loc[idx_group_out,key](*tuple(table_this_group_in.reset_index()[[key]].values[:,0]))
                        # elif apply_groups_out_df_this_group.loc[idx_group_out,key] is None:
                        #     raise ValueError('Not supported yet')
                        else:
                            apply_groups_out_df_this_group.loc[idx_group_out,key]= apply_groups_out_df.loc[idx_group_out,key]
                table_this_group_out = apply_groups_out_df_this_group

                
                dataarrays_group_in = []
                for idx_group_in,row in table_this_group_in.iterrows():
                    if table_this_group_in.index.names[0] is None: # trivial case where no group_in selection is made
                        index_dataarray = [dict(zip(table_this_group_in.columns,row))[key] for key in self.lib_dataarrays.index.names]
                    else:
                        index_dataarray = [{**dict(zip(table_this_group_in.index.names,idx_group_in)),**dict(zip(table_this_group_in.columns,row))}[key] for key in self.lib_dataarrays.index.names]

                    #index_dataarray = [{**dict(zip(table_this_group_in.index.names,idx_group_in)),**dict(zip(table_this_group_in.columns,row))}[key] for key in self.lib_dataarrays.index.names]
                    dataarrays_group_in.append(self.dataarrays[tuple(index_dataarray)])
            
                ifile = 0
                attributes_dataarrays_out = []
                for idx_group_out,row in table_this_group_out.iterrows():
                    attributes_dataarrays_out.append({})

                    if inherit_attributes:
                        if table_this_group_in.index.names[0] is None: # trivial case where no group_in selection is made
                            attributes_in = \
                                dict(zip(table_this_group_in.columns,
                                    table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)]))

                        else:
                            attributes_in = \
                                { 
                                **dict(zip(table_this_group_in.index.names,
                                    table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)].name)),
                                **dict(zip(table_this_group_in.columns,
                                    table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)]))
                                }


                        for key,value in attributes_in.items():
                            if (key not in attributes_dataarrays_out[ifile]) and \
                               ((inherit_attributes == True) or (key in inherit_attributes)) and \
                               (key not in ['absolute_path_as_cache','absolute_path_for_reading','absolute_path','path']):
                                attributes_dataarrays_out[ifile][key] = value

                    #!!
                    for key in self.lib_dataarrays.index.names:
                        if key in table_this_group_out.columns:
                            attributes_dataarrays_out[ifile][key] = row[key]
                        elif key in table_this_group_in.index.names:
                            attributes_dataarrays_out[ifile][key] = table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)].name[table_this_group_in.index.names.index(key)]
                        else:
                            attributes_dataarrays_out[ifile][key] = table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)][key]
                    for key in row.index:
                        attributes_dataarrays_out[ifile][key] = row[key]
            

                    



                        # for key in self.lib_dataarrays.columns:
                        #     if key == 'provider':
                        #         import pdb; pdb.set_trace()

                        #     if (key not in attributes_dataarrays_out[ifile]) and (inherit_attributes or (key in inherit_attributes)) and (key not in ['absolute_path','path']):
                        #         attributes_dataarrays_out[ifile][key] = {**dict(zip(table_this_group_in.index.names,table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)])),**dict(zip(table_this_group_in.columns,table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)]))}[key]
            
                    for key,value in extra_attributes.items():
                        attributes_dataarrays_out[ifile][key] = value
                    ifile += 1

                if mode in ['numpy_output_to_disk_in_chunks','numpy_output_to_disk_no_chunks']:
                    if (archive_out.file_pattern is None):
                        raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern") 
                    filenames_out = []
                    ifile = 0
                    for idx_group_out,row in table_this_group_out.iterrows():
                      filename_out = os.path.dirname(archive_out.path_pickle)+'/'+''.join(np.array(list(zip(archive_out.file_pattern.split('"')[::2],[attributes_dataarrays_out[ifile][key] for key in archive_out.file_pattern.split('"')[1::2]]+['']))).ravel())
                      if filename_out in archive_out.lib_dataarrays.absolute_path.unique():
                          raise ValueError('filename '+filename_out+ ' already exists in the output library. Consider revising the output file_pattern.') 
                      filenames_out.append(filename_out)

                      ifile += 1
            
                if mode == 'xarray':
                    print('making a temporary dataarray copy to prevent data hanging around into memory afterwards')
                    dataarrays_group_in_copy = [dataarray.copy(deep=False) for dataarray in dataarrays_group_in]
                    temp_dataarrays = func(*dataarrays_wrapper(*tuple(dataarrays_group_in_copy)))
                    #temp_dataarrays = func(*dataarrays_wrapper(*tuple(dataarrays_group_in)))

                    for idataarray,dataarray in enumerate(temp_dataarrays):
                        for key,value in attributes_dataarrays_out[idataarray].items():
                            if key == 'variable':
                                dataarray.name = value
                            else:
                                dataarray.attrs[key] = value
                        archive_out.add_dataarray(dataarray)#attributes_dataarrays_out[idataarray])
                    for idataarray in range(len(dataarrays_group_in_copy)):
                        dataarrays_group_in_copy[idataarray].close()
                    for itemp_dataarray in range(len(temp_dataarrays)):
                        temp_dataarrays[itemp_dataarray].close()

                elif mode in ['numpy_output_to_disk_in_chunks','numpy_output_to_disk_no_chunks']:

                    if mode == 'numpy_output_to_disk_in_chunks':
                        xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_group_in)),filenames_out=filenames_out,attributes = attributes_dataarrays_out, release=True, initialize_array=initialize_array,copy_coordinates=copy_coordinates,**kwargs)
                    elif mode == 'numpy_output_to_disk_no_chunks':
                        temp_dataarrays = xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_group_in)),**kwargs)
                        if type(temp_dataarrays) != tuple:
                            print('this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                            idataarray = 0
                            for key,value in attributes_dataarrays_out[idataarray].items():
                                if key not in ['variable','absolute_path_for_reading','absolute_path_as_cache','absolute_path','path']:
                                    if type(value) == bool:
                                        temp_dataarrays.attrs[key] = int(value)
                                    else:
                                        temp_dataarrays.attrs[key] = value
                                if key == 'variable':
                                    temp_dataarrays.name = value
                            #import pdb;pdb.set_trace()
                            os.system('rm '+filenames_out[idataarray])
                            if post_apply is not None:
                                post_apply(temp_dataarrays)
                            os.system('mkdir -p '+os.path.dirname(filenames_out[idataarray]))
                            temp_dataarrays.to_netcdf(filenames_out[idataarray])
                            temp_dataarrays.close()
                        else:
                            for idataarray in range(len(temp_dataarrays)):
                                for key,value in attributes_dataarrays_out[idataarray].items():
                                    if key not in ['variable','absolute_path_for_reading','absolute_path_as_cache','absolute_path','path']:
                                        temp_dataarrays[idataarray].attrs[key] = value
                                    if key == 'variable':
                                        temp_dataarrays[idataarray].name = value
            
                            for idataarray in range(len(temp_dataarrays)):
                                if post_apply is not None:
                                    post_apply(temp_dataarrays[idataarray])
                                os.system('rm '+filenames_out[idataarray])
                                os.system('mkdir -p '+os.path.dirname(filenames_out[idataarray]))
                                temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                                temp_dataarrays[idataarray].close()
            
                    for ixr_out,filename_out in enumerate(filenames_out):
                        archive_out.add_dataarray(filename_out)
                else:
                    ValueError('mode '+ mode+ ' not implemented')
                for idataarray in range(len(dataarrays_group_in)):
                    dataarrays_group_in[idataarray].close()
        if write_mode == 'create_new_archive':
            return archive_out
    def apply_func_old(self,func, xarray_function_wrapper=apply_func,dataarrays_wrapper = lambda *x: (*x,),groupby=None,apply_merge=[],apply_merge_out=[],archive_out = None,keep_in_memory_during_processing = False, inherit_attributes = False,extra_attributes={}, **kwargs):

        if (archive_out is None) and (self.path_pickle is None):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')

        temp_df = self.lib_dataarrays.reset_index() 


        if type(apply_merge) == pd.core.frame.DataFrame:
            apply_merge_df = apply_merge
        elif type(apply_merge) == list:
            apply_merge_df = pd.DataFrame(apply_merge)
        elif type(apply_merge) == dict:
            apply_merge_df = pd.DataFrame.from_dict([dict(zip(apply_merge.keys(),v)) for v in itertools.product(*apply_merge.values())])
        apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        if type(apply_merge_out) == pd.core.frame.DataFrame:
            apply_merge_out_df = apply_merge_out
        elif type(apply_merge_out) == list:
            apply_merge_out_df = pd.DataFrame(apply_merge_out)
        elif type(apply_merge_out) == dict:
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out.keys(),v)) for v in itertools.product(*apply_merge_out.values())])
        
        if len(apply_merge_out_df) == 0:
            print('creating automatic single output table')
            apply_merge_out_dict = {}
            for column in apply_merge_df.columns:
                apply_merge_out_dict[column] = ['from__'+'__'.join(apply_merge_df[column].unique())]
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out_dict.keys(),v)) for v in itertools.product(*apply_merge_out_dict.values())])
                

        apply_merge_out_index = pd.MultiIndex.from_frame(apply_merge_out_df)


        #apply_merge_pandas = pd.DataFrame([{'variable':'msshf_0001'},{'variable':'mslhf_0001','source':'cds_era5'}])
        
        #self.lib_dataarrays.index.names
        if groupby is None:
            groupby = list(self.lib_dataarrays.index.names)
            for key in apply_merge_df.columns:
                groupby = list(filter((key).__ne__, groupby))

        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        if write_mode == 'create_new_archive':
            archive_out = archive()
            archive_out.dump(path_archive_out)
        elif write_mode == 'add_to_current_archive':
            archive_out = self

        xr_outputs = {}
        for index,group in temp_df.groupby(groupby):

            group_columns = group.set_index(list(apply_merge_df.columns))
            if type(group_columns.index) == pd.core.indexes.base.Index:
                MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
                group_columns.index = MultiIndex_from_Single_Index(group_columns.index)

            all_variables_available_in_this_group = True
            if len(apply_merge_index.intersection(group_columns.index)) != len(apply_merge_index):
             all_variables_available_in_this_group = False
            # if not group_columns.loc[apply_merge_index][groupby].isnull().any().any():
            #  all_variables_available_in_this_group = False
            #  import pdb; pdb.set_trace()

            if all_variables_available_in_this_group:
                  dataarrays_for_func = []

                  for index_group,columns in group_columns.loc[apply_merge_index].iterrows():
                      index_array_dict =   {**dict(columns),**dict(zip(apply_merge_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
                      index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])

                      if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
                          print('to be implemented')
                          import pdb; pdb.set_trace()
                      else:
                          dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])

                  filenames_out = []
                  
                  attributes = []
                  ifile = 0
                  for index_group,group_columns in apply_merge_out_df.iterrows():
                      index_array_out_dict = {**dict(zip(groupby,index)),**dict(zip(apply_merge_out_df.columns,group_columns))}
                      attributes.append(index_array_out_dict)
                      index_array_out_tuple_ordered =  tuple([index_array_out_dict[key] for key in self.lib_dataarrays.index.names])


                      if inherit_attributes:
                          for key,value in dataarrays_for_func[min(len(dataarrays_for_func),ifile)].attrs.items():
                              if (key not in self.lib_dataarrays.index.names)  and \
                                 ( inherit_attributes or ((type(inherit_attributes) is list) and (key in inherit_attributes))) and \
                                 (key not in attributes[-1].keys()):
                                  attributes[-1][key] = value

                      for key,value in extra_attributes.items():
                          attributes[-1][key] = value

                      if (archive_out.file_pattern is None):
                          raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern") 
                      filenames_out.append(os.path.dirname(archive_out.path_pickle)+'/'+''.join(np.array(list(zip(archive_out.file_pattern.split('"')[::2],[attributes[-1][key] for key in archive_out.file_pattern.split('"')[1::2]]+['']))).ravel()))
                      ifile +=1

                  for ixr_out,filename_out in enumerate(filenames_out):
                      index_array_out_tuple_ordered =  tuple([attributes[ixr_out][key] for key in archive_out.lib_dataarrays.index.names])
                      if index_array_out_tuple_ordered in archive_out.dataarrays.keys():
                          self.remove(index_array_out_tuple_ordered,delete_on_disk=True)
                  

                  for filename_out in filenames_out:
                      os.system('mkdir -p '+os.path.dirname(filename_out))

                  if not keep_in_memory_during_processing:
                    xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_for_func)),filenames_out=filenames_out,attributes = attributes, release=True, **kwargs)
                  else:
                    temp_dataarrays = xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_for_func)),**kwargs)

                    if type(temp_dataarrays) != tuple:
                        print('this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                        idataarray = 0
                        for key,value in attributes[idataarray].items():
                            if key not in ['variable','absolute_path','absolute_path_as_cache','absolute_path_for_reading','path']:
                                temp_dataarrays.attrs[key] = value
                            if key == 'variable':
                                temp_dataarrays.name = value
                        #import pdb;pdb.set_trace()
                        os.system('rm '+filenames_out[idataarray])
                        temp_dataarrays.to_netcdf(filenames_out[idataarray])
                        temp_dataarrays.close()
                    else:
                        for idataarray in range(len(temp_dataarrays)):
                            for key,value in attributes[idataarray].items():
                                if key not in ['variable','absolute_path_as_cache','absolute_path_for_reading','absolute_path','path']:
                                    temp_dataarrays[idataarray].attrs[key] = value
                                if key == 'variable':
                                    temp_dataarrays[idataarray].name = value

                        for idataarray in range(len(temp_dataarrays)):
                            os.system('rm '+filenames_out[idataarray])
                            temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                            temp_dataarrays[idataarray].close()

                  for ixr_out,filename_out in enumerate(filenames_out):
                      archive_out.add_dataarray(filename_out)
        if write_mode == 'create_new_archive':
            return archive_out



#             if type(group_columns_out.index) == pd.core.indexes.base.Index:
#                 MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
#                 group_columns_out.index = MultiIndex_from_Single_Index(group_columns_out.index)
#            
#             # not_all_arrays_available_in_this_group = group_columns.loc[apply_merge_out_index][groupby].isnull().any().any()
#             #       dataarrays_for_func = []
# 
#             import pdb;pdb.set_trace()
#             for index_group,columns in group_columns_out.loc[apply_merge_out_index].iterrows():
#                 index_array_dict =   {**dict(columns),**dict(zip(apply_merge_out_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
#                 index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])
#                 # dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])
# 




                #group_columns.loc[apply_merge_index]
            

        # temp_df.groupby(self.lib_dataarrays.index.names
        # if kwargs.keys() != 1:
        #     raise ValueError('length different from 1 is not allowed')


    def update(self,
               library_path=None,
               query=None,
               force_overwrite_dataarrays = False,
               force_overwrite_pickle=False,
               extra_attributes={},
               dump_floating_dataarrays=False,**kwargs):

        """
            perform quick attribute parameter updates and/or dump unsynced information to disc on request.
        """
        if library_path is not None:
            if library_path[-4:] == '.pkl':
                lib_basename = os.path.basename(os.path.abspath(library_path))
                lib_dirname = os.path.dirname(os.path.abspath(library_path))
            elif library_path[-1:] == '/':
                lib_basename = 'master.pkl'
                lib_dirname = os.path.abspath(library_path)
            else:
                raise ValueError('Please provide a path that either ends on ".pkl" (indicating specific picklefile) or "/" (full archive dump at master.pkl)')


            if os.path.isfile(lib_dirname+'/'+lib_basename) and (not force_overwrite_pickle):
                raise IOError('pickle file exists. Please use force_overwrite_pickle = True, or specify the full pickle name as the library_path.')

            # when the self.path_pickle exists, it is assumed that this file needs to be updated
            self.path_pickle = lib_dirname+'/'+lib_basename
        # else:
        #     lib_dirname = os.path.dirname(self.path_pickle)


        # if self.path_pickle is None:
        #     raise ValueError("I can't track the pickle name. Please specify a pkl file or directory with library_path")

        for key,value in kwargs.items():
            if key not in self.settings_keys:
                raise ValueError('"'+key+'" is not a known setting.')
            else:
                self.__dict__[key] = value

        if query is not None:
            read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()

        if len(extra_attributes) > 0:
            for idx,row in read_lib_dataarrays.iterrows():

                # # this should become part of a save procedure updating the attributes of the netcdf on disc
                # if ('absolute_path' in row.keys()) and ('ncvariable' in row.keys()):
                #     variable=idx[self.lib_dataarrays.index.names.index('variable')]
                #     self.remove(idx)
                #     ncin = nc4.open_dataset(row['absolute_path'])
                #     for attribute_key,attribute_value in extra_attributes.items():
                #         ncin[idx[self.lib_dataarrays.index.names.index('variable')]].setncattr(attribute_key,attribute_value)
                #     ncin.close()
                #     self.add_dataarray(row['absolute_path'],variable,ncvariable=row['ncvariable'])
                # else:
                dataarray_temp = self.dataarrays[idx]

                attributes_temp = {**dict(zip(read_lib_dataarrays.index.names,idx)),**row}
                for attribute_key,attribute_value in extra_attributes.items():
                    attributes_temp[attribute_key] = attribute_value

                # extra_attributes_plus_path = extra_attributes.copy()
                # if ( ('path' in row.keys()) and (type(row['path']) == str)):
                #     extra_attributes_plus_path['path'] =row['path']
                # if ( ('absolute_path' in row.keys()) and (type(row['absolute_path']) == str)):
                #     extra_attributes_plus_path['absolute_path'] =row['absolute_path']

                self.remove(idx)
                self.add_dataarray(dataarray_temp,**attributes_temp)

        # for key,value in extra_attributes.items():
        #     self.lib_dataarrays.loc[read_lib_dataarrays.index][key] = value
        
        if 'path_pickle' in self.__dict__.keys():
            os.system('mkdir -p '+os.path.dirname(self.path_pickle))

        read_lib_dataarrays = self.lib_dataarrays.copy()

        for idx,columns in read_lib_dataarrays.iterrows():
            if ( ('absolute_path' not in columns.keys()) or (type(columns['absolute_path']) != str)) or \
               ( ('path' not in columns.keys()) or (type(columns['path']) != str)):

                if dump_floating_dataarrays:
                    #parse filename according to file_pattern
                    if 'path_pickle' not in self.__dict__.keys():
                        raise ValueError ('self.path_pickle is not set')
                    fnout = os.path.dirname(self.path_pickle)+'/'+''.join(np.array(list(zip(self.file_pattern.split('"')[::2],[{**dict(zip(self.lib_dataarrays.index.names,idx)),**columns}[key] for key in self.file_pattern.split('"')[1::2]]+['']))).ravel())
                    print("File pointer for ",idx," is not known, so I'm dumping a new file under ",fnout)
                    #fnout = self.lib_dataarrays.loc[idx]['absolute_path']
                    if (not force_overwrite_dataarrays) and (os.path.isfile(fnout)):
                        raise IOError(fnout+' exists. Use force_overwrite_dataarrays to overwrite file')
                    os.system('mkdir -p '+os.path.dirname(fnout))
                    # self.dataarrays[idx].attrs['absolute_path'] = fnout
                    for key,value in dict(columns).items():
                        self.dataarrays[idx]

                    for key,value in dict(columns).items():
                        if key not in ['variable','absolute_path','absolute_path_for_reading','absolute_path_as_cache','path']:
                            if type(value) == bool:
                                self.dataarrays[idx].attrs[key] = int(value)
                            else:
                                self.dataarrays[idx].attrs[key] = value
                        if key == 'variable':
                            self.dataarrays[idx].name = value
 
                    os.system('rm '+fnout)
                    self.dataarrays[idx].to_netcdf(fnout);print('file written to: '+fnout)
                    self.remove(idx)
                    self.add_dataarray(fnout)
                    #self.dataarrays[idx]

                    # key = 'path'
                    # if key not in self.lib_dataarrays.columns:
                    #     self.lib_dataarrays[key] = ''
                    # self.lib_dataarrays.loc[idx]['path'] = './'

                    # note that path and absolute_path are not written to the netcdf file above, but it is available virtually for convenience
                    #self.lib_dataarrays['absolute_path'].loc[idx] = fnout
                #self.dataarrays[idx].attrs['path'] = self.lib_dataarrays.loc[idx]['path']
            else:

                print("Assuming variable for ",idx," exists in file "+columns['absolute_path'])
                if 'path' not in self.lib_dataarrays.columns:
                    self.lib_dataarrays['path'] = None

            if 'path_pickle' in self.__dict__.keys():
                if ((columns['absolute_path'] is not None) and (type(columns['absolute_path']) is str)):
                    if 'path' not in self.lib_dataarrays.columns:
                        self.lib_dataarrays['path'] = None
                    self.lib_dataarrays['path'].loc[idx] =  os.path.relpath(columns['absolute_path'],os.path.dirname(self.path_pickle))
                    print("relative file path to "+os.path.dirname(self.path_pickle)+" is "+self.lib_dataarrays['path'].loc[idx])
                #os.path.commonprefix([columns['absolute_path'],lib_dirname])
                elif ((columns['path'] is not None) and (type(columns['path']) is str)):
                    if 'absolute_path' not in self.lib_dataarrays.columns:
                        self.lib_dataarrays['absolute_path'] = None
                    self.lib_dataarrays['absolute_path'].loc[idx] =  os.path.dirname(self.path_pickle)+'/'+columns['path']

                    if ((columns['absolute_path_for_reading'] is None) or (type(columns['absolute_path_for_reading']) is not str)):
                        if 'absolute_path_for_reading' not in self.lib_dataarrays.columns:
                            self.lib_dataarrays['absolute_path_for_reading'] = None 
                        self.lib_dataarrays['absolute_path_for_reading'].loc[idx] =  os.path.dirname(self.path_pickle)+'/'+columns['path']
                    #print("absolute file path to "+os.path.dirname(self.path_pickle)+" is "+self.lib_dataarrays['path'].loc[idx])
        
        if ('path_pickle' in self.__dict__.keys()):
            self.lib_dataarrays.to_pickle(self.path_pickle)
            with open(self.path_pickle+'.yaml','w') as file:
                yaml.dump([[key,self.__dict__[key]] for key in self.settings_keys],file)

    def load(
            self,
            path,
            path_settings = None,
            #file_pattern = lambda columns: columns['variable']+'_'+columns['source']+'_'+columns['time']+'_'+columns['space']+'.nc',
            file_pattern = None,
            query= None,
            extra_attributes={},
            skip_unavailable=False,
            add_file_pattern_matches=False,
            release_dataarray_pointer =False,
            cache_to_tempdir=False,
            cache_to_ram=False,
            **kwargs):


        if type(path).__name__ == 'list':
            # eg., -> files_wildcard = '*_*_*_*.nc'
            print('Guessing files from file list...(this procedure may need revision)')
            #allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
            filenames = path
            for filename in filenames:
                
                self.add_dataarray(xr.open_dataarray(filename),absolute_path=filename,skip_unavailable=False,release_dataarray_pointer =True,cache_to_tempdir=False,**extra_attributes)

        # elif os.path.isfile(path):
        #     print('pkl file '+path+' detected. Listing files from there.' )
        #     lib_dataarrays_temp = pd.read_pickle(path)
        #     if query is not None:
        #         print('performing query subselection: '+query)
        #         lib_dataarrays_temp = lib_dataarrays_temp.query(query,engine='python')
        #     for index,dataarray in lib_dataarrays_temp.iterrows():
        #         allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
        #         if dataarray.path[0] == '/':
        #             absolute_file_path = dataarray.path
        #         else:
        #             absolute_file_path = os.path.dirname(path)+'/'+dataarray.path
        #         allkwargs['absolute_path'] = absolute_file_path
        #         if os.path.isfile(absolute_file_path):
        #             print('adding '+absolute_file_path+' with additional attributes: ',allkwargs)
        #             try: 
        #                 xrin = xr.open_dataarray(absolute_file_path)
        #             except ValueError:
        #                 # try to open the variable out of dataset
        #                 variable = index[0]
        #                 xrin = xr.open_dataset(absolute_file_path)[variable]
        #             self.add_dataarray(xrin,release_dataarray_pointer =False,**allkwargs)

        #         else:
        #             print('Warning. Could not add file '+absolute_file_path+' because it does not exist.')

        # elif os.path.isdir(path):
        #     print('Guessing files from directory...')
        #     self.lib_dirname = path
        #     self.lib_basename = 'master.pkl'

        elif type(path) == str:
            if path[-4:] == '.pkl':
                lib_basename = os.path.basename(os.path.abspath(path))
                lib_dirname = os.path.dirname(os.path.abspath(path))
            elif path[-1:] == '/':
                lib_basename = 'master.pkl'
                lib_dirname = os.path.abspath(path)
            else:
                raise ValueError('Please provide a path that either ends on ".pkl" (indicating specific picklefile) or "/" (full archive dump at master.pkl)')
        else:
            raise IOError ('path '+path+ ' does not exist.')
#         os.system('mkdir -p '+path)

        # file_pattern = '"variable"_"columns"_"time"_"space".nc'


        temp_path_pickle = lib_dirname+'/'+lib_basename



        
        print('apply settings according to yaml file and kwargs')
        if path_settings is None:
            path_settings = temp_path_pickle+'.yaml'
        elif not os.path.isfile(path_settings):
            raise IOError('Settings file '+path_settings+ ' not found.')

        if os.path.isfile(path_settings):
            print('settings file found')
            with open(path_settings) as file:
                for key,value in yaml.load(file):
                    if key in self.settings_keys:
                        self.__dict__[key] = value 

        for key,value in kwargs.items():
            if key not in self.settings_keys:
                raise ValueError('"'+key+'" is not a known setting.')
            else:
                self.__dict__[key] = value


        # if file_pattern is not None:
        #     print('Overriding settings "file_pattern" with ', file_pattern,' from arguments')
        #     self.file_pattern = file_pattern

        
        print('reading the dataarrays from the pickle file')

        if type(query) == str:
            read_lib_dataarrays = pd.read_pickle(temp_path_pickle).query(query,engine='python')
        elif query is not None:
            read_lib_dataarrays = query(pd.read_pickle(temp_path_pickle))
        else:
            #query is None
            read_lib_dataarrays = pd.read_pickle(temp_path_pickle)

        for index,columns in read_lib_dataarrays.iterrows():
            absolute_path = None

            if 'path' in read_lib_dataarrays.columns:
                absolute_path = lib_dirname+'/'+columns['path']
            elif 'absolute_path' in read_lib_dataarrays.columns:
                absolute_path = columns['absolute_path']
                columns['path'] =  os.path.relpath(columns['absolute_path'],os.path.dirname(self.path_pickle))

            if (absolute_path is not None) and (absolute_path not in self.lib_dataarrays.absolute_path):
                #if index[0] == 'mslhf_0001':
                print('Opening file : '+absolute_path)
                self.add_dataarray(absolute_path,skip_unavailable=skip_unavailable,release_dataarray_pointer =True,cache_to_tempdir=False,cache_to_ram=cache_to_ram,**({**dict(zip(read_lib_dataarrays.index.names,index)),**columns}),**extra_attributes)

        if add_file_pattern_matches and (self.file_pattern is not None):
            files_wildcard = lib_dirname+'/'+''.join(np.array(list(zip(self.file_pattern.split('"')[::2],['*']*len(self.file_pattern.split('"')[1::2])+['']))).ravel())
            print('file_pattern is '+self.file_pattern+' and add_file_pattern_matches == True, so scanning and adding files that match the wildcard: ',files_wildcard+' that are not in the library yet')
            # eg., -> files_wildcard = '*_*_*_*.nc'
            filenames = glob.glob(files_wildcard)
            for filename in filenames:
                if filename not in self.lib_dataarrays.absolute_path:
                    path = os.path.relpath(filename,os.path.dirname(temp_path_pickle))
                    print('Opening file : '+filename)
                    self.add_dataarray(filename,skip_unavailable=skip_unavailable, release_dataarray_pointer = True, cache_to_tempdir=False,path=path,cache_to_ram=cache_to_ram,**extra_attributes)
        self.path_pickle = temp_path_pickle
        
        # import pdb; pdb.set_trace()
        if type(query) == str:
            read_lib_dataarrays = self.lib_dataarrays.query(query,engine='python').copy()
        elif (query is not None):
            read_lib_dataarrays = query(self.lib_dataarrays)
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()





        for idx,columns in self.lib_dataarrays.iterrows():
            self.remove(idx)
        for idx,columns in read_lib_dataarrays.iterrows():
            absolute_path = None
            # import pdb; pdb.set_trace()
            if 'path' in read_lib_dataarrays.columns:
                absolute_path = lib_dirname+'/'+columns['path']
            elif 'absolute_path' in read_lib_dataarrays.columns:
                absolute_path = columns['absolute_path']

            if (absolute_path is not None) and (absolute_path not in self.lib_dataarrays.absolute_path):
                #if index[0] == 'mslhf_0001':
                print('Opening file : '+absolute_path)
                self.add_dataarray(absolute_path,skip_unavailable=skip_unavailable,release_dataarray_pointer =release_dataarray_pointer,cache_to_tempdir=cache_to_tempdir,cache_to_ram=cache_to_ram,**({**dict(zip(read_lib_dataarrays.index.names,idx)),**columns}),**extra_attributes)

            


    # def dataarrays_merge_apply(self,attrs,function):
    #     if len(attrs.keys()) > 1:
    #         print('not tested, please check')
    #         import pdb; pdb.set_trace()
    #     
    #     index_keys_groupby = []
    #     index_keys_nongroupby = []
    #     for key in self.lib_dataarrays.index.names:
    #         if key in attrs.keys():
    #             index_keys_nongroupby.append(key)
    #         else:
    #             index_keys_groupby.append(key)
    #     lfirst = True

    #     archive_out = archive()
    #     for index_groupby,group_lib_dataarrays in  self.lib_dataarrays.groupby(index_keys_groupby):
    #         group_dataarrays = [self.dataarrays[key] for key in group_lib_dataarrays.index]
    #         dataarray_out = function(group_dataarrays)
    #         for column in group_lib_dataarrays.columns:
    #             # if lib_dataarrays[column].unique() > 0:
    #             # aparently uniform attributes are taken over, so we can detect from the destination attributies
    #             if (column not in dataarray_out.attrs.keys()) and (column != 'path'):
    #                 dataarray_out.coords[column] = (attrs.keys(), group_lib_dataarrays[column].values)
    #         for key,value in attrs.items():
    #             dataarray_out.attrs[key] = value
    #         archive_out.add_dataarray(dataarray_out)
    #     return archive_out

    def dataarrays_merge(self,attrs):
        if len(attrs.keys()) > 1:
            print('not tested, please check')
            import pdb; pdb.set_trace()
        
        index_keys_groupby = []
        index_keys_nongroupby = []
        for key in self.lib_dataarrays.index.names:
            if key in attrs.keys():
                index_keys_nongroupby.append(key)
            else:
                index_keys_groupby.append(key)
        lfirst = True

        archive_out = archive()
        for index_groupby,group_lib_dataarrays in  self.lib_dataarrays.groupby(index_keys_groupby):
            group_dataarrays = [self.dataarrays[key] for key in group_lib_dataarrays.index]
            dataarray_out = xr.concat(group_dataarrays,dim=group_lib_dataarrays.loc[index_groupby].index) 
            for column in group_lib_dataarrays.columns:
                # if lib_dataarrays[column].unique() > 0:
                # aparently uniform attributes are taken over, so we can detect from the destination attributies
                if (column not in dataarray_out.attrs.keys()) and (column != 'path'):
                    dataarray_out.coords[column] = (attrs.keys(), group_lib_dataarrays[column].values)
            for key,value in attrs.items():
                dataarray_out.attrs[key] = value
            archive_out.add_dataarray(dataarray_out)
        return archive_out
            # group_dataarrays = self.lib_dataarrays.loc[index_groupby]
            # xr.concat(group_dataarrays.values(),dim=group_dataarrays.index)

            # if lfirst:
            #     dataarray_out = self.dataarrays



    #    for key,value in merges.items():

    #        self.lib_dataarrays.groupby:


    # def merge_time_in_space(self,time,datetimes,space):
    #     archive_temp = self.dataarrays_apply(lambda x: x.reindex({'time':datetimes}),attrs={'time':time})

    #         archive_out = archive()
    #         import pdb; pdb.set_trace()


        


        # for key in 
        # self.lib_dataarrays

        # self.timeframes = pd.DataFrame(index=empty_multiindex(['timeframe','grid','source','variable']),columns=['model','experiment','ensemble_member','start','end','start_clip','end_clip']).iloc[1:]
        # self.timeframes_pages = pd.DataFrame(index = empty_multiindex(['timeframe','grid','source','variable','start','end']),columns = list(self.pages.columns)+['start_clip','end_clip']).iloc[1:]


    # def add_source(self,source = 'gsod_historical',pathfile = ''):
    #     if source == 'ghcn_historical':    
    #         lib_temp = pd.read_fwf(pathfile,names=['locid','lat','lon','variable','yearstart','yearend'])
    #         self.lookupvar[source] = {'pr_dailysum':'PRCP'}
    #         for var,sourcevar in self.lookupvar[source].items():
    #             
    #             lib_temp['variable'].loc[lib_temp['variable'] == sourcevar] = var    # = lib_temp.var.str.rename(columns={0:self.lookupvar[source][seriesvar]})
    #     else:
    #         raise ValueError ('Source '+source+' not supported.')
    #         
    #     lib_temp['source'] = source
    #     lib_temp = lib_temp.set_index(['locid','source','variable'])
    #     
    #     self.lib = pd.concat([self.lib,lib_temp])    


