import os
import copy

import json

from csrank.experiments.dbconnection_modified import ModifiedDBConnector


def combine_with_elements(previous_dict, index, combos):
    print("previous: {}, index: {}".format(previous_dict, index))
    if index < len(combos) - 1:
        new_dicts = []
        for element in combos[index]:
            print("add element {} to dict {}".format(element, previous_dict))
            new_dict = copy.deepcopy(previous_dict)
            new_dict.update(element)
            print("new dict", new_dict)
            new_dicts.extend(combine_with_elements(new_dict, index + 1, combos))
        print("returning", new_dicts)
        return new_dicts
    else:
        new_dicts = []
        for element in combos[index]:
            new_dict = copy.deepcopy(previous_dict)
            new_dict.update(element)
            new_dicts.append(new_dict)
        print("returning", new_dicts)
        return new_dicts


def gen_jobs(configuration_file_path):
    # here there are given certain arguments and this method generates a combination of them
    # reading them from json file would probably be ideal
    if os.path.isfile(configuration_file_path):
        configuration_file = open(configuration_file_path, "r")
        config_str = configuration_file.read().replace('\n', '')
        config = json.loads(config_str)
    else:
        raise ValueError('File does not exist for the configuration of the database')

    combos = []
    for key in config:
        # for every key and value, we need to add all their combos to a final result
        value = config[key]
        if isinstance(value, list):
            combos.append(value)
        else:
            combos.append([{key: value}])

    # now go through combos and add them to a final job
    return combine_with_elements(previous_dict={}, index=0, combos=combos)


job = {'resources': {'CPUs': 4, 'GPUs': 0},
       'dataset': 'TSP',
       'dataset_params': {'include_id': 'False', 'n_objects_train': 20, 'n_objects_test': 20,
                          'n_train_instances': 500, 'n_test_instances': 2, 'filename': 'cities.csv'},
       'fold_id': 1,
       'n_inner_folds': 0,
       'learning_problem': 'OBJECT_RANKING_TSP',
       'seed': 1234,
       'learner_name': 'fate_choice',
       'learner_params': {
           'loss_function': 'losses.tsp_dist_matrix_loss_wrapper',
           'loss_function_requires_x_values': 'True'
        },
       'learner_fit_params': {},
       'use_hp': "False",
       'hp_iterations': 0,
       'hp_ranges': {},
       'hp_fit_params': {},
       'duration': '60M',
       'time_out_eval': '10M',
       'results_table_name': 'results_test'}

config_file_path = "config/db.json"

connector = ModifiedDBConnector(config_file_path, table_jobs="jobs_test")
# connector.insert_new_job(job)

jobs = gen_jobs("configuration.json")
for combo_job in jobs:
    connector.insert_new_job(combo_job)
