import yaml
try:
    from yaml import CLoader as yaml_Loader, CDumper as yaml_Dumper
except ImportError:
    from yaml import Loader as yaml_Loader, yaml_Dumper


import logging
from . import archive,collection
import os
import datetime as dt
from time import sleep, ctime
import numpy as np
import tempfile
from ast import literal_eval
from subprocess import Popen, PIPE
import random


broker_global_settings_defaults = {
    'conda_environment_export': '/tmp/anaconda/bin',
    'dummy': 'False',
    'reset_archive': 0,
    'reset_lock':0,
    'reset_history':0,
    'force_recalculate':0,
    'delay':5,
    'datetime_start_chain': '__undefinied__'
}

class broker (object):
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            self.__dict__[key] = value
        self.global_settings()#default settings

    def propagate_arguments_to_parents(self):
        logging.info('BEGIN --- propagate arguments to parent processes --- ')
        if type(self.provides) is not list:
            for ibroker_requires, broker_requires in enumerate(self.requires):
                for key in list(broker_requires.keys()):
                    if (broker_requires[key] is None) or (type(broker_requires[key]) is int):
                        if key in self.provides.keys():
                            if type(self.provides[key]) is list:
                                iselect = (0 if (broker_requires[key] is None) else broker_requires[key])
                                self.requires[ibroker_requires][key] = self.provides[key][iselect]
                            else:
                                self.requires[ibroker_requires][key] = self.provides[key]
                        # elif key in vars(args).keys():
                        #     self.requires[ibroker_requires][key] = vars(args)[key]
                        else:
                            del self.requires[ibroker_requires][key]
                    elif type(broker_requires[key]) is type(lambda x:x):
                        pass_parameters = []
                        for ibroker_provides, broker_provides in enumerate(self.provides):
                            pass_parameters.append(broker_provides[key])
                        self.requires[ibroker_requires][key] = self.requires[ibroker_requires][key](pass_parameters)
        logging.info('END --- propagate arguments to parent processes --- ')

    def global_settings(self,**kwargs):

        for setting in broker_global_settings_defaults.keys():
            if (setting in kwargs.keys()) and (kwargs[setting] is not None):
                self.__dict__[setting] = type(broker_global_settings_defaults[setting])(kwargs[setting])
            else:
                self.__dict__[setting] = broker_global_settings_defaults[setting]

        setting = 'datetime_start_chain'
        if self.__dict__[setting] == '__undefinied__':
            self.__dict__[setting] = str(dt.datetime.now()).replace(' ','_')


    def flush_history_file(self,history_dict, history_filename,history_ok=True):
        if (self.dummy != 'True') and (history_ok == True):
            if not os.path.isdir(os.path.dirname(history_filename)):
                os.makedirs(os.path.dirname(history_filename))
            with open(history_filename, 'w') as history_file:
                dump = yaml.dump(history_dict,Dumper=yaml_Dumper)
                history_file.write(dump)

    def retrieve_input(self,unique_archive_paths=True,query_return_dict=False,debug=False):
        logging.info('--- BEGIN Collecting or generating coarse input data -- ')
        for ibroker_requires, broker_requires in enumerate(self.requires):

            request_parent = self.requires[ibroker_requires].copy()
            for key, item in list(request_parent.items()):
                if type(item) is type(lambda x: x):
                    del request_parent[key]
                if key in ['archive', 'process','inherit']:
                    del request_parent[key]

            if 'process' in broker_requires:
                parent_execute = [
                    './lib/pynacolada/pynacolada/launch_python_conda.sh',
                    self.conda_environment_export,
                    broker_requires['process']
                ]
                # parent_execute = ['source', '/projects/C3S_EUBiodiversity/bio_load.sh','&&', 'python',broker_requires['process']]
                parent_arguments = []

                for key, value in request_parent.items():
                    # if key != 'spaces':
                    parent_arguments.append('--' + key)
                    # if type(value) not in [int,bool]:
                    #     parent_execute.append('"'+str(value)+'"')
                    # else:
                    parent_arguments.append(str(value))

                logging.info('propagate environment settings')
                arguments_propagate_to_parent = ['conda_environment_export', 'dummy']
                for key in arguments_propagate_to_parent:
                    parent_arguments.append('--' + key)
                    parent_arguments.append(self.__dict__[key])
                    if type(parent_arguments[-1]) is str:
                        parent_arguments[-1] = '"'+parent_arguments[-1]+'"'


                arguments_propagate_reduce_to_parent = [
                    'reset_archive',
                    'reset_lock',
                    'reset_history',
                    'force_recalculate'
                ]
                for key in arguments_propagate_reduce_to_parent:
                    parent_execute.append('--' + key)
                    parent_execute.append(str(max(int(self.__dict__[key])-1, 0)))
                parent_execute.append('--delay')
                parent_execute.append(str(self.delay))
                parent_execute.append('--datetime_start_chain')
                parent_execute.append(str(self.datetime_start_chain))

                # p = Popen(parent_arguments , stdout=PIPE, stderr=PIPE)
                for iarg, arg in enumerate(parent_arguments):
                    if arg == "":
                        parent_arguments[iarg] = '_empty_'
                    else:
                        parent_arguments[iarg] = parent_arguments[iarg].replace('""','"')

                for iarg, arg in enumerate(parent_execute):
                    if arg == "":
                        parent_execute[iarg] = '_empty_'
                    else:
                        parent_execute[iarg] = parent_execute[iarg].replace('""','"')

                self.requires[ibroker_requires]['process_arguments'] = str(request_parent)


                history_filename = self.requires[ibroker_requires]['root'] + '/requests/' + os.path.basename(self.requires[ibroker_requires][ 'process'] + '_history.yaml')
                if (not os.path.isfile(history_filename)):
                    history_dict = {} #reset_history
                else:
                    # try:
                        with open(history_filename, 'r') as history_file:
                            try:
                                history_dict = yaml.load(history_file,Loader=yaml_Loader)
                                if history_dict is None:
                                    history_dict = {}
                            except:
                                logging.warning('read history file failed. starting from scratch')
                                history_dict = {}
                    # except:
                    #     logging.warning('reading history file failed: '+history_filename)
                    #     history_dict = {}


                # if debug == True:
                #     import pdb; pdb.set_trace()

                if (((self.reset_history - 1) > 0)) or ((self.reset_archive - 1) > 0):
                    if (self.requires[ibroker_requires]['process_arguments'] in history_dict.keys()):
                        del history_dict[self.requires[ibroker_requires]['process_arguments']]
                    self.flush_history_file(history_dict,history_filename)

                # if ((self.reset_archive - 1) > 0):
                #     history_dict = {}
                #     self.flush_history_file(history_dict,history_filename)

                if (self.requires[ibroker_requires]['process_arguments'] in history_dict.keys()):
                    self.requires[ibroker_requires]['return_from_history'] = history_dict[self.requires[ibroker_requires]['process_arguments']]['return_from_subprocess']
                else:
                    tempdir=self.requires[ibroker_requires]['root'] + '/log/'
                    if not os.path.isdir(tempdir):
                        os.makedirs(tempdir)
                    tempbasename = tempfile.mktemp(
                        prefix=broker_requires['process'].split('/')[-1] + '_',
                        dir=tempdir)
                    self.requires[ibroker_requires]['stderr'] = open(tempbasename + '_e.log', 'w')
                    self.requires[ibroker_requires]['stdout'] = open(tempbasename + '_o.log', 'w')
                    logging.info('- stdout: ' + self.requires[ibroker_requires]['stdout'].name)
                    logging.info('- stderr: ' + self.requires[ibroker_requires]['stderr'].name)

                    self.requires[ibroker_requires]['executing_subprocess'] = \
                        Popen(
                            parent_execute + parent_arguments,
                            # ' '.join(parent_execute), #' "' + ('" "'.join(parent_arguments))+'" ',
                            # (' '.join(parent_execute) + '"' + '" "'.join(parent_arguments)),
                            stdout=self.requires[ibroker_requires]['stdout'],
                            stderr=self.requires[ibroker_requires]['stderr'],
                            )
            #make process-specific file. if that one exists. then we can assume that lock file is also created. then we continue
            delay_random = self.delay * (1+random.random())
            logging.info('delaying after start subprocess for '+str(delay_random)+' seconds')
            sleep(delay_random)

        for ibroker_requires,broker_requires in list(enumerate(self.requires)):
            if  ('executing_subprocess' in broker_requires.keys()) or ('return_from_history' in broker_requires.keys()):
                history_filename = self.requires[ibroker_requires]['root'] + '/requests/' + os.path.basename(self.requires[ibroker_requires][ 'process'] + '_history.yaml')

                history_ok = True
                if (not os.path.isfile(history_filename)):
                    history_dict = {}
                else:
                    try:
                        with open(history_filename, 'r') as history_file:
                            history_dict = yaml.load(history_file,Loader=yaml_Loader)
                            if history_dict is None:
                                history_dict = {}
                    except:
                        logging.warning('reading history file failed: '+history_filename)
                        history_dict = {}
                        #history_ok = False

                if 'return_from_history' in self.requires[ibroker_requires].keys():
                    return_from_subprocess = \
                       self.requires[ibroker_requires]['return_from_history']


                    if self.requires[ibroker_requires]['process_arguments'] in history_dict.keys():
                        history_dict[self.requires[ibroker_requires]['process_arguments']]['number_of_requests'] += 1
                    else:
                        history_dict[self.requires[ibroker_requires]['process_arguments']] = { \
                            'return_from_subprocess': return_from_subprocess,
                            'number_of_requests': 1
                        }
                else:
                    logging.info('waiting for subprocess: '+broker_requires['stdout'].name)
                    logging.info('                        '+broker_requires['stderr'].name)
                    broker_requires['executing_subprocess'].wait()
                    self.requires[ibroker_requires]['stderr'].close()
                    self.requires[ibroker_requires]['stdout'].close()
                    logging.info('finished    subprocess: '+broker_requires['stdout'].name)
                    logging.info('                        '+broker_requires['stderr'].name)
                    return_from_subprocess = open(self.requires[ibroker_requires]['stdout'].name,'r').readlines()[-1][:-1]

                    if self.requires[ibroker_requires]['process_arguments'] in history_dict.keys():
                        del history_dict[self.requires[ibroker_requires]['process_arguments']]

                    print('retrieving from parent process '+self.requires[ibroker_requires]['stdout'].name+': '+return_from_subprocess)
                    history_dict[self.requires[ibroker_requires]['process_arguments']] = { \
                        'return_from_subprocess': return_from_subprocess,
                        'number_of_requests': 1
                    }
                    logging.info('return statement from subprocess: '+return_from_subprocess)


                if debug == True:
                    import pdb; pdb.set_trace()
                return_from_subprocess_eval = literal_eval(return_from_subprocess)

                if (return_from_subprocess_eval) == None:
                    self.requires[ibroker_requires]['disable'] = True
                elif type(return_from_subprocess_eval) == list:
                    for ireturn_requires,return_requires in enumerate(return_from_subprocess_eval):
                        self.requires.append(return_requires)
                        # if debug ==True:
                        #     import pdb; pdb.set_trace()
                        for key in ['root','process', 'executing_subprocess', 'stderr', 'stdout','process_arguments','inherit']:
                            if key in broker_requires.keys():
                                self.requires[-1][key] = broker_requires[key]
                    self.requires[ibroker_requires]['disable'] = True
                elif type(return_from_subprocess_eval) == dict:
                    for key,value in return_from_subprocess_eval.items():
                        if (type(value) is list) and (len(value) <= 1):
                            return_from_subprocess_eval[key] = return_from_subprocess_eval[key][0]
                        self.requires[ibroker_requires].update(return_from_subprocess_eval)
                else:
                    raise IOError('subprocess return type not implemented')

                logging.info('content history: '+str(history_dict))
                logging.info('writing to history file: '+history_filename)
                self.flush_history_file(history_dict, history_filename,history_ok=True)
                #broker['requires'][ibroker_requires]['archive'] = pcd.archive( args.root_requires + '/' + broker_requires['archive'])
            #make process-specific file. if that one exists. then we can assume that lock file is also created. then we continue
            delay_random = self.delay * (1+random.random())
            logging.info('delaying retrieval for '+str(delay_random)+' seconds')
            sleep(delay_random)
        if (type(self.provides) == list):
            # for ibroker_provides, broker_provides in enumerate(self.provides):
            #     if ('archive' in broker_provides.keys()) and (type(broker_provides['archive']) is type(lambda x:x)):
            #         values_input = []
            #         for item in self.requires:
            #             if ('disable' not in item.keys()):
            #                 if ('archive' in item.keys()):
            #                     values_input.append(item['archive'])
            #                 else:
            #                     values_input.append(None)
            #         if len(values_input) > 0:
            #             self.provides[ibroker_provides]['archive'] = broker_provides['archive'](values_input)
            #         else:
            #             self.provides[ibroker_provides]['archive'] = None  # broker_provides['archive'](values_input)
            if type(self.requires) == list:
                logging.info('propagating keywords from "requires" to "provides"')

                for ibroker_provides, broker_provides in enumerate(self.provides):
                    for key in self.provides[ibroker_provides].keys():
                        if key in ['archive',]: #those who are not handled by archive.apply_func 
                            if type(self.provides[ibroker_provides][key]).__name__ == 'function':
                                values_input = []
                                for ibroker_requires, broker_requires in list(enumerate(self.requires)):
                                    if ('disable' not in self.requires[ibroker_requires].keys()):
                                        if (key in self.requires[ibroker_requires].keys()) and (self.requires[ibroker_requires][key] is not None):
                                            values_input.append(self.requires[ibroker_requires][key])
                                        # else:
                                        #     values_input.append(None)
                                if len(values_input) > 0:
                                        self.provides[ibroker_provides][key] = self.provides[ibroker_provides][key](*tuple(values_input))
                                else:
                                    logging.critical('no value for provides - function found.')
                                    self.provides[ibroker_provides][key] = None  # broker_provides['archive'](values_input)

                icount = 0
                for ibroker_requires,broker_requires in enumerate(self.requires):
                    if ('disable' not in self.requires[ibroker_requires].keys()) and \
                     not (('inherit' in self.requires[ibroker_requires].keys()) and (self.requires[ibroker_requires]['inherit'] == False)):
                        for key in self.requires[ibroker_requires].keys():
                            if key not in [
                                'process',
                                'root',
                                'process_arguments',
                                'stderr',
                                'stdout',
                                'executing_subprocess',
                                'return_from_history',
                                'disable',
                                'inherit']:

                                if (len(self.provides) >= (icount+1)) and (key not in self.provides[icount].keys()):
                                    self.provides[icount][key] = self.requires[ibroker_requires][key]

                                for ibroker_provides, broker_provides in reversed(list(enumerate(self.provides))):
                                    if (key not in self.provides[ibroker_provides].keys()):
                                        #import pdb; pdb.set_trace()
                                        self.provides[ibroker_provides][key] = self.requires[ibroker_requires][key]

                        icount += 1
        else:
            broker_provides = self.provides
            if ('archive' in broker_provides) and (type(broker_provides['archive']) is type(lambda x: x)):
                values_input = []
                for item in self.requires:
                    if 'disable' not in item.keys():
                        if 'archive' in item.keys():
                            values_input.append(item['archive'])
                        else:
                            values_input.append(None)
                if len(values_input) > 0:
                    self.provides['archive'] = broker_provides['archive'](values_input)
                else:
                    self.provides['archive'] = None #broker_provides['archive'](values_input)

        if type(self.provides ) == list:
            if self.provides[0]['archive'] is not None:
                archive_out_filename =self.provides[0]['root'] + '/' + self.provides[0]['archive']
            else:
                archive_out_filename = None
        else:
            if self.provides['archive'] is not None:
                archive_out_filename = self.provides['root'] + '/' + self.provides['archive']
            else:
                archive_out_filename = None

        if (self.reset_archive > 0) \
                and \
                (archive_out_filename is not None) and \
                os.path.isfile(archive_out_filename) and \
                (str(dt.datetime.strptime(ctime(os.path.getctime(archive_out_filename)), "%a %b %d %H:%M:%S %Y")).replace(' ','_') < self.datetime_start_chain)\
                :

            logging.info('Resetting archive of current broker')
            if type(self.provides) is list:
                for broker_current in self.provides:
                    if 'archive' in broker_current.keys():
                        if type(broker_current['archive']) is str:
                            archive_out = archive(broker_current['root'] + '/' + broker_current['archive'])

                            archive_out.remove(reset_lib=True)
                            archive_out.close()
                        elif type(broker_current['archive']) is archive:
                            broker_current['archive'].remove(reset_lib=True)
                            broker_current['archive'].close()
                        sleep(0.1)
            else:
                if 'archive' in self.provides.keys():
                    if type(self.provides['archive']) is str:
                        archive_out = archive(self.provides['root'] + '/' + self.provides['archive'])
                        archive_out.remove(reset_lib=True)
                        archive_out.close()
                    elif type(self.provides['archive']) is archive:
                        self.provides['archive'].remove(records=True)
                        self.provides['archive'].close()

        # if debug == True:
        #     import pdb; pdb.set_trace()

        archive_collection = []
        archive_collection_paths = []


        for broker_requires in self.requires:
            if ('archive' in broker_requires.keys()):
                full_path = os.path.realpath(broker_requires['root'] + '/' + broker_requires['archive'])
                if ((full_path not in archive_collection_paths)) or (unique_archive_paths == False):
                    archive_collection_paths.append(full_path)
                    if query_return_dict:
                        if unique_archive_paths == True:
                            raise ValueError(' for option query_return_dict, please turn on unique_archive_paths')
                        return_dict = broker_requires.copy()
                        for key in ['process_arguments','archive','root','process']:
                            return_dict.pop(key)
                    else:
                        return_dict = None

                    archive_collection.append(archive(full_path,query_dict=return_dict,debug=debug))


        

        self.parent_collection = collection(archive_collection)#[archive(full_path,debug=debug) for full_path in archive_collection_paths])
        delay_random = self.delay * (1+random.random())
        logging.info('delaying after input retrieve for '+str(delay_random)+' seconds')
        sleep(delay_random)

        logging.info('--- END Collecting or generating coarse input data --- ')


    def subset_requires(self,subset_variables,key='variable'):
        # this should be transformed into a query-like procedure
        broker_requires_new = []
        for variable in subset_variables:
            first_variable_found = False
            for ibroker_requires,broker_requires in enumerate(self.requires):
                if (not first_variable_found) and (key in broker_requires) and (broker_requires[key] == variable):
                    broker_requires_new.append(broker_requires)
                    first_variable_found = True
            # if not first_variable_found:
            #     raise ValueError ('value '+variable+'not found in self.requires')
        self.requires = broker_requires_new

    def get_return_request( self, sources=None,return_exclude_keys=[], return_also_non_index_keys = True, debug=False):

        if debug==True:
            import pdb; pdb.set_trace()
        if type(self.provides) is list:

            return_request = list()
            for igroups_out in range(len(self.provides)):
                return_request.append(dict())
                for key in self.provides[igroups_out].keys():
                    if (key not in ['root','chain']) and (key not in return_exclude_keys) and ((key in ['archive','variable','time','space','source']) or return_also_non_index_keys) :
                        if type(self.provides[igroups_out][key]) is not type(lambda x:x):
                            return_request[igroups_out][key] = self.provides[igroups_out][key]
                        else:
                            values_input = []
                            for item in sources:
                                if 'disable' not in item.keys():
                                    if key in item.keys():
                                        values_input.append(item[key])
                                    # else:
                                    #     values_input.append(None)
                            if len(values_input) > 0:
                                return_request[igroups_out][key] = self.provides[igroups_out][key](*tuple(values_input))
                        # if type(return_request[igroups_out][key]) is str:
                        #     return_request[igroups_out][key].replace('"','')

            if debug==True:
                import pdb; pdb.set_trace()

            common_keys = list(return_request[0].keys())

            for item in return_request:
                for key in common_keys:
                    if key not in item.keys():
                        common_keys.remove(key)

            return_request_common_keys = []
            for item in return_request:
                return_request_common_keys_item = {}
                for key in common_keys:
                    return_request_common_keys_item[key] = item[key]
                if return_request_common_keys_item not in return_request_common_keys:
                    return_request_common_keys.append(return_request_common_keys_item)

            for key in return_request[0].keys():
                if key not in return_request_common_keys[0].keys():
                    return_request_common_keys[0][key] = return_request[0][key]
            return_request = return_request_common_keys
            if debug==True:
                import pdb; pdb.set_trace()



        else:
            return_request = dict()
            for key in self.provides.keys():
                if key not in ['root', 'chain']:
                    if type(self.provides[key]) == list:
                        return_request[key] = []
                        if debug == True:
                            import pdb; pdb.set_trace()
                        for ireturn_request in range(len(self.provides[key])):
                            if (type(self.provides[key][ireturn_request]) != type(lambda x:x)) and \
                                    (key not in return_exclude_keys) and \
                                    ((key in ['archive','variable','time','space','source']) or return_also_non_index_keys) \
                                    :
                                return_request[key].append(self.provides[key][ireturn_request])
                                # if type(return_request[key][-1]) == str:
                                #     return_request[key][-1] = return_request[key][-1].replace('"','')

                        if len(return_request[key]) == 0:
                            del return_request[key]
                    else:
                        if debug == True:
                            import pdb; pdb.set_trace()
                        if (type(self.provides[key]) != type(lambda x: x)) and \
                                (key not in return_exclude_keys):
                            return_request[key] = self.provides[key]
        if debug==True:
            import pdb; pdb.set_trace()
        return str(return_request).replace(' ','')


    def apply_func(
            self,
            apply_groups_out_extra=None,
            query=None,
            sources=None,
            return_exclude_keys=[],
            return_also_non_index_keys = True,
            debug=False,
            reset_archive_after = False,
            *args,
            **kwargs):

        if sources == None:
            sources = self.requires
        #for ibroker_provides, broker_provides in enumerate(self.provides):
        if debug==True:
            import pdb; pdb.set_trace()
        if type(self.provides ) == list:
            archive_out_filename =self.provides[0]['root'] + '/' + self.provides[0]['archive']
        else:
            archive_out_filename = self.provides['root'] + '/' + self.provides['archive']
        lockfile = archive_out_filename + '_lock'

        if return_also_non_index_keys == True:
            logging.warning('apply_func is set to return also non index keys to parent function (return_also_non_index_keys == True). In the future, this will not be the ddefault behaviour anymore (default will be return_also_non_index_keys == False)')



        if (self.reset_lock > 0) and \
                os.path.isfile(lockfile) and \
                (str(dt.datetime.strptime(ctime(os.path.getctime(lockfile)), "%a %b %d %H:%M:%S %Y")).replace(' ','_') < self.datetime_start_chain):
            os.system('rm ' + lockfile)

        while os.path.isfile(lockfile):
            logging.warning('archive locked with "' + lockfile + '" . Waiting for another process to release it.')
            sleep(5+ random.random()*10.)

            #another nested check to be really safe
            while os.path.isfile(lockfile):
                logging.warning('archive locked with "' + lockfile + '" . Waiting for another process to release it.')
                sleep(5+ random.random()*10.)

        requests_parents = [broker_requires.copy() for broker_requires in sources]

        for irequest_parent,request_parent in list(reversed(list(enumerate(requests_parents)))):
            if ('disable' in request_parent.keys()) \
                    and (requests_parents[irequest_parent]['disable'] == True):
                del requests_parents[irequest_parent]

        parent_collection_lib_dataarrays = self.parent_collection.get_lib_dataarrays()

        for irequest_parent,request_parent in list(reversed(list(enumerate(requests_parents)))):
            for key,value in list(requests_parents[irequest_parent].items()):

                if \
                     ((key in ['archive','process','executing_subprocess','stderr','stdout','process_arguments']) or \
                     (((key not in parent_collection_lib_dataarrays.columns)) and \
                       (key not in parent_collection_lib_dataarrays.index.names)) or \
                             (type(value) is type(lambda x: x))):
                 del requests_parents[irequest_parent][key]

        if type(self.provides) is list:
            apply_groups_out = list()
            for igroups_out in range(len(self.provides)):
                apply_groups_out.append(dict())
                for key in self.provides[igroups_out].keys():
                    if key not in ['archive','root','chain']:
                        apply_groups_out[igroups_out][key] = self.provides[igroups_out][key]
        else:
            apply_groups_out = dict()
            for key in self.provides.keys():
                if key not in ['archive', 'root', 'chain']:
                    if type(self.provides[key]) is list:
                        apply_groups_out[key] = []
                        for iapply_groups_out in range(len(self.provides[key])):
                            apply_groups_out[key].append(self.provides[key][iapply_groups_out])
                    else:
                        apply_groups_out[key] = self.provides[key]

        # for irequest_parent,request_parent in list(enumerate(requests_parents)):
        #     for key in list(requests_parents[irequest_parent]):
        #         if key not in self.parent_collection.get_lib_dataarrays().index.names:# ['variable','source','time','space']:
        #             del requests_parents[irequest_parent][key]
        if debug == True:
            import pdb; pdb.set_trace()

        logging.info('start apply_func on parent_collection')
        if self.dummy != 'True':

            CMD = 'mkdir -p '+os.path.dirname(lockfile)
            logging.debug('executing: '+CMD); os.system(CMD)
            CMD = 'touch '+lockfile
            logging.debug('executing: '+CMD); os.system(CMD)

            self.parent_collection.apply_func(
                self.operator,
                query=query,
                apply_groups_in = requests_parents,
                apply_groups_out=apply_groups_out,
                archive_out = archive_out_filename,
                add_archive_out_to_collection = False, # TODO: should be False (default) but need to check why that gives problems.
                delay = self.delay,
                **self.operator_properties,
                **kwargs)
 
        CMD = 'rm '+lockfile
        logging.debug('removing lockfile by "'+CMD+'"'); os.system(CMD)

        logging.info('output pkl file: '+  archive_out_filename)
        logging.info('output directory: '+  os.path.dirname(archive_out_filename))
        if reset_archive_after == True:
            reset_archive_after()



        logging.info('end apply_func on parent_collection')



        logging.info('Dumping return_request as last line in stdout being used for processes depending on it')

        return self.get_return_request(sources=sources ,return_exclude_keys = return_exclude_keys,return_also_non_index_keys = return_also_non_index_keys,debug=debug)

    def reset_archive_after(self):

            for irequires,requires in enumerate(self.requires):
               history_filename = self.requires[iself_requires]['root'] + '/requests/' + os.path.basename(self.requires[iself_requires][ 'process'] + '_history.yaml')
               history_dict = {}
               self.flush_history_file(history_dict,history_filename)

            logging.info('removing parent files')
            for archive in self.parent_collection.archives:
                archive.remove(query=("variable_key == '" + variable_key + "'"), reset_lib=True)




        # for apply_groups_out in apply_groups_out_list:
        #         for key in apply_groups_out.keys():
        #             if type(apply_groups_out[key]) != type(lambda x:x):
        #                 if key not in return_request.keys():
        #                     return_request[key] = []
        #                 return_request[key].append(apply_groups_out[key])
        # for key,list_values in list(return_request.items()):
        #     list_values_unique = list(np.unique(list_values))
        #     if len(list_values_unique) == 1 :
        #        return_request[key] = list_values_unique

        #     return_request['archive'] = self.provides[0]['archive']


