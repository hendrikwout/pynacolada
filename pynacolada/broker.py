import yaml
import logging
from . import archive,collection
import os
import numpy as np
import tempfile
from ast import literal_eval
from subprocess import Popen, PIPE


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
        self.global_settings_defaults = {
            'conda_environment_export': 'blabla',
            'dummy': 'False',
            'reset_archive': 0,
            'reset_lock':0,
            'force_recalculate':0,
        }

        for setting in self.global_settings_defaults.keys():
            if setting in kwargs.keys():
                self.__dict__[setting] = kwargs[setting]

    def retrieve_input(self,debug=False):
        logging.info('--- BEGIN Collecting or generating coarse input data -- ')
        for ibroker_requires, broker_requires in enumerate(self.requires):

            request_parent = self.requires[ibroker_requires].copy()
            for key, item in list(request_parent.items()):
                if type(item) is type(lambda x: x):
                    del request_parent[key]
                if key in ['archive', 'process','root']:
                    del request_parent[key]

            if 'process' in broker_requires:
                parent_execute = [
                    './bin/launch_python_conda.sh',
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
                    'force_recalculate'
                ]
                for key in arguments_propagate_reduce_to_parent:
                    parent_execute.append('--' + key)
                    parent_execute.append(str(max(self.__dict__[key]-1, 0)))

                # p = Popen(parent_arguments , stdout=PIPE, stderr=PIPE)
                for iarg, arg in enumerate(parent_arguments):
                    if arg == "":
                        parent_arguments[iarg] = '_empty_'
                for iarg, arg in enumerate(parent_execute):
                    if arg == "":
                        parent_execute[iarg] = '_empty_'

                self.requires[ibroker_requires]['process_arguments'] = str(request_parent)

                history_filename = self.requires[ibroker_requires]['root'] + '/requests/' + self.requires[ibroker_requires][ 'process'] + '_history.yaml'
                if (not os.path.isfile(history_filename)) or ((self.reset_archive - 1) > 0):
                    history_dict = {} #reset_history
                else:
                    with open(history_filename, 'r') as history_file:
                        history_dict = yaml.load(history_file)
                        if history_dict is None:
                            history_dict = {}

                if (self.requires[ibroker_requires]['process_arguments'] in history_dict.keys()) and \
                        not (((self.force_recalculate -1)> 0) or ((self.reset_archive - 1) > 0)):
                    self.requires[ibroker_requires]['executing_subprocess'] = 'from_history'
                else:
                    tempdir=self.requires[ibroker_requires]['root'] + '/log/'
                    if not os.path.isdir(tempdir):
                        os.mkdir(tempdir)
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
        if debug == True:
            import pdb; pdb.set_trace()

        for ibroker_requires,broker_requires in list(enumerate(self.requires)):
            if  ('executing_subprocess' in broker_requires.keys()):
                history_filename = self.requires[ibroker_requires]['root'] + '/requests/' + os.path.basename(self.requires[ibroker_requires][ 'process'] + '_history.yaml')
                if not os.path.isfile(history_filename):
                    history_dict = {}
                else:
                    with open(history_filename, 'r') as history_file:
                        history_dict = yaml.load(history_file)
                        if history_dict is None:
                            history_dict = {}

                if (self.requires[ibroker_requires]['executing_subprocess'] == 'from_history'):
                    return_from_subprocess = \
                            history_dict[self.requires[ibroker_requires]['process_arguments']]['return_from_subprocess']
                else:
                    broker_requires['executing_subprocess'].wait()
                    self.requires[ibroker_requires]['stderr'].close()
                    self.requires[ibroker_requires]['stdout'].close()
                    return_from_subprocess = open(self.requires[ibroker_requires]['stdout'].name,'r').readlines()[-1][:-1]
                    print('retrieving from parent process '+self.requires[ibroker_requires]['stdout'].name+': '+return_from_subprocess)
                    if debug == True:
                        import pdb; pdb.set_trace()
                    history_dict[self.requires[ibroker_requires]['process_arguments']] = { \
                        'return_from_subprocess': return_from_subprocess,
                        'number_of_requests': 0
                    }

                history_dict[self.requires[ibroker_requires]['process_arguments']]['number_of_requests']  += 1

                if not os.path.isdir(os.path.dirname(history_filename)):
                    os.mkdir(os.path.dirname(history_filename))
                with open(history_filename,'w') as history_file:
                    dump = yaml.dump(history_dict, default_flow_style = False)
                    history_file.write( dump )

                return_from_subprocess_eval = literal_eval(return_from_subprocess)

                if type(return_from_subprocess_eval) == list:
                    for ireturn_requires,return_requires in enumerate(return_from_subprocess_eval):
                        self.requires.append(return_requires)
                        if debug ==True:
                            import pdb; pdb.set_trace()
                        for key in ['root','process', 'executing_subprocess', 'stderr', 'stdout','process_arguments']:
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

                #broker['requires'][ibroker_requires]['archive'] = pcd.archive( args.root_requires + '/' + broker_requires['archive'])
        if type(self.provides) == list:
            for ibroker_provides, broker_provides in enumerate(self.provides):
                if ('archive' in broker_provides) and (type(broker_provides['archive']) is type(lambda x:x)):
                    values_input = []
                    for item in self.requires:
                        if 'archive' in item.keys():
                            values_input.append(item['archive'])
                        else:
                            values_input.append(None)
                    self.provides[ibroker_provides]['archive'] = broker_provides['archive'](values_input)
        else:
            broker_provides = self.provides
            if ('archive' in broker_provides) and (type(broker_provides['archive']) is type(lambda x: x)):
                values_input = [item['archive'] for item in self.requires]
                self.provides['archive'] = broker_provides['archive'](values_input)

        if self.reset_archive > 0 :
            logging.info('Resetting archive of current broker')
            if type(self.provides) is list:
                for broker_current in self.provides:
                    if 'archive' in broker_current.keys():
                        if type(broker_current['archive']) is str:
                            archive_out = archive(broker_current['root'] + '/' + broker_current['archive'])
                            archive_out.remove(records=True)
                            archive_out.close()
                        elif type(broker_current['archive']) is archive:
                            broker_current['archive'].remove(records=True)
                            broker_current['archive'].close()
            else:
                if 'archive' in self.provides.keys():
                    if type(self.provides['archive']) is str:
                        archive_out = archive(self.provides['root'] + '/' + self.provides['archive'])
                        archive_out.remove(records=True)
                        archive_out.close()
                    elif type(self.provides['archive']) is archive:
                        self.provides['archive'].remove(records=True)
                        self.provides['archive'].close()
        if debug == True:
            import pdb; pdb.set_trace()

        archive_collection = []
        for broker_requires in self.requires:
            if 'archive' in broker_requires.keys():
                archive_collection.append(archive(broker_requires['archive']))

        self.parent_collection = collection(archive_collection)

        logging.info('--- END Collecting or generating coarse input data --- ')

    def apply_func(self,apply_groups_out_extra=None,sources=None,return_exclude_keys=[],debug=False,*args,**kwargs):
        if sources == None:
            sources = self.requires
        #for ibroker_provides, broker_provides in enumerate(self.provides):
        if type(self.provides ) == list:
            archive_out_filename =self.provides[0]['root'] + '/' + self.provides[0]['archive']
        else:
            archive_out_filename = self.provides['root'] + '/' + self.provides['archive']
        lockfile = archive_out_filename + '_lock'
        if self.reset_lock > 0:
            os.system('rm ' + lockfile)
        while os.path.isfile(lockfile):
            logging.info('archive locked with "' + lockfile + '" . Waiting for another process to release it.')
            sleep(10)

        requests_parents = [broker_requires.copy() for broker_requires in sources]

        for irequest_parent,request_parent in list(enumerate(requests_parents)):
            if ('disable' in request_parent.keys()) \
                    and (request_parent['disable'] == True):
                del requests_parents[irequest_parent]
            else:
                for key,value in list(request_parent.items()):
                    if \
                            (key in ['archive','process','executing_subprocess','stderr','stdout','process_arguments']) or \
                                (key not in self.parent_collection.get_lib_dataarrays().columns and \
                            key not in self.parent_collection.get_lib_dataarrays().index.names) or \
                            type(value) is type(lambda x: x):
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

        self.parent_collection.apply_func(
            self.operator,
            apply_groups_in = requests_parents,
            apply_groups_out=apply_groups_out,
            archive_out = archive_out_filename,
            *args,
            **self.operator_properties,
            **kwargs)
        os.system('rm '+lockfile)

        logging.info('Dumping return_request as last line in stdout being used for processes depending on it')

        if type(self.provides) is list:

            return_request = list()
            for igroups_out in range(len(self.provides)):
                return_request.append(dict())
                for key in self.provides[igroups_out].keys():
                    if key not in ['root','chain']:
                        if type(self.provides[igroups_out][key]) is not type(lambda x:x):
                            return_request[igroups_out][key] = self.provides[igroups_out][key]
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
                                    (key not in return_exclude_keys):
                                return_request[key].append(self.provides[key][ireturn_request])
                        if len(return_request[key]) == 0:
                            del return_request[key]
                    else:
                        if debug == True:
                            import pdb; pdb.set_trace()
                        if (type(self.provides[key]) != type(lambda x: x)) and \
                               (key not in return_exclude_keys):
                            return_request[key] = self.provides[key]




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
        return str(return_request).replace(' ','')
